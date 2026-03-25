"""
训练主脚本 — LMAFusion（含 AGM 自适应自引导）

用法:
  # 基础训练
  python train.py --data_path ./data --epoch 50 --batch_size 8

  # AGM 自引导训练（推荐）
  python train.py --data_path ./data_csmr --epoch 50 --batch_size 8 --use_agm

  # 断点续训
  python train.py --data_path ./data --epoch 50 --is_resume ./model/best.pth
"""

import argparse
import datetime
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

from dataset import FusionDataset
from losses import fusion_loss, VGGPerceptualLoss
from net import LMAFusion
from utils import save_model, load_model, AverageMeter, setup_logger


# ── AGM 辅助函数 ──────────────────────────────────────────────

def cc(ir: torch.Tensor, vis: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    """
    Cross-Correlation 质量评估：衡量 fused 与 IR/VIS 的相关性均值。
    值越高说明 fused 同时保留了 IR 和 VIS 的信息。
    """
    A = ir.squeeze()
    B = vis.squeeze()
    F = fused.squeeze()
    batch = A.shape[0] if A.dim() == 3 else 1
    if A.dim() == 2:
        A, B, F = A.unsqueeze(0), B.unsqueeze(0), F.unsqueeze(0)

    c = 0.0
    for i in range(batch):
        a, b, f = A[i] * 255, B[i] * 255, F[i] * 255
        rAF = torch.sum((a - a.mean()) * (f - f.mean())) / (
            torch.sqrt(torch.sum((a - a.mean()) ** 2) * torch.sum((f - f.mean()) ** 2)) + 1e-8)
        rBF = torch.sum((b - b.mean()) * (f - f.mean())) / (
            torch.sqrt(torch.sum((b - b.mean()) ** 2) * torch.sum((f - f.mean()) ** 2)) + 1e-8)
        c += (rAF + rBF) / 2
    return c / batch


def agm_self(ir: torch.Tensor, vis: torch.Tensor,
             fused: torch.Tensor, i2: torch.Tensor):
    """
    自适应自引导模块（纯自引导版本）。

    比较 fused 与 I2（模型历史最佳输出）的质量，
    返回引导目标、自适应权重和是否需要更新 I2 的标记。
    """
    with torch.no_grad():
        fused_d = Variable(fused.data.clone(), requires_grad=False)
        i2_d = Variable(i2.data.clone(), requires_grad=False)
        ir_d = Variable(ir.data.clone(), requires_grad=False)
        vis_d = Variable(vis.data.clone(), requires_grad=False)

        w1 = cc(ir_d, vis_d, fused_d)
        w3 = cc(ir_d, vis_d, i2_d)

        if torch.isnan(w3):
            w3 = torch.tensor(0.0)
        if w1 < 0 or w3 <= 0:
            w1 = w1 + 1
            w3 = w3 + 1

        w = 3 * w3 / (w1 + 1e-8)
        flag = w1 >= w3

    return i2, w.item(), flag


def update_i2_fullres(model, data_dir: str, device, i2_dir: str, epoch: int):
    """
    每个 epoch 结束后，用全分辨率图像跑一遍模型，更新 I2。
    epoch==0 无条件保存；之后仅在 fused 质量超过已有 I2 时更新。
    """
    model.eval()
    ir_dir = Path(data_dir) / 'ir'
    vis_dir = Path(data_dir) / 'vis'
    if not vis_dir.exists():
        vis_dir = Path(data_dir) / 'vi'

    ir_files = sorted(list(ir_dir.glob('*.jpg')) + list(ir_dir.glob('*.png')))
    cnt_update = 0

    with torch.no_grad():
        for ir_path in ir_files:
            name = ir_path.name
            vis_path = vis_dir / name
            if not vis_path.exists():
                continue

            ir_buf = np.fromfile(str(ir_path), dtype=np.uint8)
            ir_img = cv2.imdecode(ir_buf, cv2.IMREAD_GRAYSCALE)
            if ir_img is None:
                continue
            ir_img = ir_img.astype(np.float32) / 255.0

            vis_buf = np.fromfile(str(vis_path), dtype=np.uint8)
            vis_img = cv2.imdecode(vis_buf, cv2.IMREAD_GRAYSCALE)
            if vis_img is None:
                continue
            vis_img = vis_img.astype(np.float32) / 255.0

            h, w = ir_img.shape[:2]
            vh, vw = vis_img.shape[:2]
            if (vh, vw) != (h, w):
                vis_img = cv2.resize(vis_img, (w, h))

            ir_t = torch.from_numpy(ir_img).unsqueeze(0).unsqueeze(0).to(device)
            vis_t = torch.from_numpy(vis_img).unsqueeze(0).unsqueeze(0).to(device)

            fused = model(ir_t, vis_t)

            stem = ir_path.stem
            i2_path = Path(i2_dir) / f"{stem}.jpg"

            should_save = (epoch == 0)
            if not should_save and i2_path.exists():
                i2_buf = np.fromfile(str(i2_path), dtype=np.uint8)
                i2_img = cv2.imdecode(i2_buf, cv2.IMREAD_GRAYSCALE)
                if i2_img is not None:
                    i2_img = i2_img.astype(np.float32) / 255.0
                    if i2_img.shape[:2] != (h, w):
                        i2_img = cv2.resize(i2_img, (w, h))
                    i2_t = torch.from_numpy(i2_img).unsqueeze(0).unsqueeze(0).to(device)
                    w_new = cc(ir_t, vis_t, fused)
                    w_old = cc(ir_t, vis_t, i2_t)
                    should_save = (w_new >= w_old)
                else:
                    should_save = True
            elif not should_save:
                should_save = True

            if should_save:
                fused_np = (fused[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(i2_path), fused_np)
                cnt_update += 1

    print(f"  I2 全分辨率更新: {cnt_update}/{len(ir_files)} 张")
    model.train()


# ── 训练 / 验证函数 ──────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, writer, epoch, opts,
                    vgg_loss=None):
    model.train()
    meter_total = AverageMeter('loss')
    meter_int = AverageMeter('int')
    meter_ssim = AverageMeter('ssim')
    meter_sf = AverageMeter('sf')
    meter_align = AverageMeter('align')
    meter_perceptual = AverageMeter('perc')
    meter_guidance = AverageMeter('guid')

    cnt_update = 0
    start = time.time()

    for it, (ir, vis, i2, names) in enumerate(loader):
        ir = ir.to(device)
        vis = vis.to(device)
        i2 = i2.to(device)

        fused, aligned_ir, offset = model(ir, vis)

        # AGM: 计算引导权重（epoch 0 无引导）
        gd_weight = 0.0
        gd_img = None
        flag = True
        if opts.use_agm and epoch > 0:
            gd_img, gd_weight, flag = agm_self(aligned_ir, vis, fused, i2)
            if flag:
                cnt_update += 1

        total_loss, detail = fusion_loss(
            fused, aligned_ir, vis,
            lambda_int=opts.lambda_int,
            lambda_ssim=opts.lambda_ssim,
            lambda_sf=opts.lambda_sf,
            lambda_ir_sal=opts.lambda_ir_sal,
            lambda_tv=10.0,
            offset=offset,
            vgg_loss=vgg_loss,
            lambda_perceptual=opts.lambda_perceptual,
            use_align=opts.use_align,
            guidance_img=gd_img,
            guidance_weight=gd_weight,
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        meter_total.update(detail['total'])
        meter_int.update(detail['intensity'])
        meter_ssim.update(detail['ssim'])
        meter_sf.update(detail['sf'])
        meter_align.update(detail['align'])
        meter_perceptual.update(detail['perceptual'])
        meter_guidance.update(detail['guidance'])

        global_step = epoch * len(loader) + it
        if it % opts.log_freq == 0:
            writer.add_scalar('train/loss', detail['total'], global_step)
            writer.add_scalar('train/sf', detail['sf'], global_step)
            writer.add_scalar('train/guidance', detail['guidance'], global_step)
            elapsed = time.time() - start
            gd_str = f"Guid={detail['guidance']:.4f} w={gd_weight:.2f}" if opts.use_agm else ""
            print(
                f"  Ep[{epoch}/{opts.epoch}] It[{it}/{len(loader)}] "
                f"Loss={detail['total']:.4f} "
                f"Int={detail['intensity']:.4f} "
                f"SSIM={detail['ssim']:.4f} "
                f"SF={detail['sf']:.4f} "
                f"Perc={detail['perceptual']:.4f} "
                f"{gd_str} [{elapsed:.1f}s]"
            )

    if opts.use_agm:
        print(f"  AGM: {cnt_update}/{len(loader)} batches updated I2")

    return meter_total.avg


def validate(model, loader, device, opts, vgg_loss=None):
    model.eval()
    meter = AverageMeter('val_loss')
    with torch.no_grad():
        for ir, vis, _, _ in loader:
            ir, vis = ir.to(device), vis.to(device)
            fused = model(ir, vis)
            total_loss, _ = fusion_loss(
                fused, ir, vis,
                lambda_int=opts.lambda_int,
                lambda_ssim=opts.lambda_ssim,
                lambda_sf=opts.lambda_sf,
                lambda_ir_sal=opts.lambda_ir_sal,
                lambda_tv=10.0,
                offset=None,
                vgg_loss=vgg_loss,
                lambda_perceptual=opts.lambda_perceptual,
                use_align=opts.use_align,
            )
            meter.update(total_loss.item())
    return meter.avg


# ── 主函数 ────────────────────────────────────────────────────

def main(opts):
    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # I2 目录初始化
    if opts.use_agm:
        i2_dir = opts.i2_dir
        os.makedirs(i2_dir, exist_ok=True)
        print(f"AGM 自引导已开启，I2 目录: {i2_dir}")

    # 数据集
    val_split = 'val' if os.path.isdir(os.path.join(opts.data_path, 'val')) else 'test'
    train_set = FusionDataset(
        os.path.join(opts.data_path, 'train'),
        patch_size=opts.patch_size, augment=True,
        i2_dir=opts.i2_dir if opts.use_agm else None,
    )
    val_set = FusionDataset(
        os.path.join(opts.data_path, val_split),
        patch_size=opts.patch_size, augment=False,
    )
    train_loader = DataLoader(train_set, batch_size=opts.batch_size,
                              shuffle=True, num_workers=opts.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers)

    # 模型
    model = LMAFusion(
        base_ch=opts.base_ch,
        d_state=opts.d_state,
        exchange_p=opts.exchange_p,
        use_align=opts.use_align,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params / 1e3:.1f}K)")

    optimizer = Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=opts.epoch, eta_min=opts.lr * 0.01)

    log_dir = os.path.join(opts.log_dir, opts.name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger = setup_logger(os.path.join(log_dir, 'train.log'))
    logger.info(f"模型参数量: {total_params:,}")
    if opts.use_agm:
        logger.info("AGM 自适应自引导已开启")

    start_epoch = 0
    best_val_loss = float('inf')
    if opts.is_resume:
        start_epoch, best_val_loss = load_model(opts.is_resume, model, optimizer, device)
        print(f"从 Epoch {start_epoch} 继续训练，历史最佳 loss={best_val_loss:.4f}")

    os.makedirs(opts.model_path, exist_ok=True)

    vgg_loss = VGGPerceptualLoss().to(device) if opts.lambda_perceptual > 0 else None

    train_data_dir = os.path.join(opts.data_path, 'train')

    for epoch in range(start_epoch, opts.epoch):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, writer, epoch, opts,
            vgg_loss=vgg_loss)

        if opts.use_agm and opts.i2_dir:
            update_i2_fullres(model, train_data_dir, device, opts.i2_dir, epoch)

        val_loss = validate(model, val_loader, device, opts, vgg_loss=vgg_loss)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        msg = (f"Epoch [{epoch}/{opts.epoch}] "
               f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} LR={lr_now:.2e}")
        logger.info(msg)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('lr', lr_now, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(opts.model_path, 'best.pth')
            save_model(best_path, epoch, best_val_loss, model, optimizer)
            print(f"  [BEST] Saved model -> {best_path}  (val_loss={val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(opts.model_path, f'epoch_{epoch}.pth')
            save_model(ckpt_path, epoch, val_loss, model, optimizer)

    writer.close()
    print("训练完成！")


def parse_opt():
    parser = argparse.ArgumentParser(description='LMAFusion 训练脚本')

    # 数据
    parser.add_argument('--data_path', type=str, default='./data', help='数据根目录')
    parser.add_argument('--patch_size', type=int, default=128, help='训练 patch 大小')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')

    # 模型
    parser.add_argument('--base_ch', type=int, default=16, help='基础通道数')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba SSM 状态维度')
    parser.add_argument('--exchange_p', type=float, default=0.5, help='通道交换比例')

    # 训练
    parser.add_argument('--epoch', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--lambda_int', type=float, default=1.0, help='强度损失权重')
    parser.add_argument('--lambda_ssim', type=float, default=1.0, help='SSIM 损失权重')
    parser.add_argument('--lambda_sf', type=float, default=1.0, help='SF 纹理损失权重')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='感知损失权重')
    parser.add_argument('--lambda_ir_sal', type=float, default=0.0, help='红外显著性增强损失权重')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU 编号')
    parser.add_argument('--is_resume', type=str, default=None, help='断点续训权重路径')

    # AGM 自引导
    parser.add_argument('--use_agm', action='store_true', help='开启 AGM 自适应自引导训练')
    parser.add_argument('--i2_dir', type=str, default='./data_csmr/train/i2', help='I2 引导图像目录')

    # 输出
    parser.add_argument('--name', type=str, default='LMAFusion', help='实验名称')
    parser.add_argument('--model_path', type=str, default='./model', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard 日志目录')
    parser.add_argument('--log_freq', type=int, default=20, help='打印频率（iteration）')
    parser.add_argument('--use_align', action='store_true', help='开启DCN形变对齐')

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)
