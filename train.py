"""
训练主脚本 — LMAFusion

用法:
  # 首次运行（准备数据集后）
  python train.py --data_path ./data --epoch 50 --batch_size 8

  # 断点续训
  python train.py --data_path ./data --epoch 50 --is_resume ./model/best.pth

数据准备（只需执行一次）:
  from dataset import DroneDatasetPreparer
  DroneDatasetPreparer.prepare(
      ir_src  = r'C:/.../.../train_by_yolo_IR/images/train',
      vis_src = r'C:/.../.../train_by_yolo_RGB/images/train',
      output_dir = './data'
  )
"""

import argparse
import datetime
import os
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False
    class SummaryWriter:  # dummy writer
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

from dataset import FusionDataset
from losses import fusion_loss, VGGPerceptualLoss
from net import LMAFusion
from utils import save_model, load_model, AverageMeter, setup_logger


def train_one_epoch(model, loader, optimizer, device, writer, epoch, opts, vgg_loss=None):
    model.train()
    meter_total = AverageMeter('loss')
    meter_int = AverageMeter('int')
    meter_ssim = AverageMeter('ssim')
    meter_grad = AverageMeter('grad')
    meter_align = AverageMeter('align')
    meter_perceptual = AverageMeter('perc')

    start = time.time()
    for it, (ir, vis, _) in enumerate(loader):
        ir = ir.to(device)
        vis = vis.to(device)

        fused, aligned_ir, offset = model(ir, vis)

        # 注意：此处必须使用 aligned_ir 去算下游损失
        # 因为用原图算会强制网络输出具有双影的错误融合结果！
        total_loss, detail = fusion_loss(
            fused, aligned_ir, vis,  
            lambda_int=opts.lambda_int,
            lambda_ssim=opts.lambda_ssim,
            lambda_grad=opts.lambda_grad,
            lambda_tv=10.0,
            offset=offset,
            vgg_loss=vgg_loss,
            lambda_perceptual=opts.lambda_perceptual,
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        meter_total.update(detail['total'])
        meter_int.update(detail['intensity'])
        meter_ssim.update(detail['ssim'])
        meter_grad.update(detail['gradient'])
        meter_align.update(detail['align'])
        meter_perceptual.update(detail['perceptual'])

        global_step = epoch * len(loader) + it
        if it % opts.log_freq == 0:
            writer.add_scalar('train/loss', detail['total'], global_step)
            writer.add_scalar('train/intensity', detail['intensity'], global_step)
            writer.add_scalar('train/ssim', detail['ssim'], global_step)
            writer.add_scalar('train/gradient', detail['gradient'], global_step)
            writer.add_scalar('train/align', detail['align'], global_step)
            writer.add_scalar('train/perceptual', detail['perceptual'], global_step)
            elapsed = time.time() - start
            print(
                f"  Ep[{epoch}/{opts.epoch}] It[{it}/{len(loader)}] "
                f"Loss={detail['total']:.4f} "
                f"Int={detail['intensity']:.4f} "
                f"SSIM={detail['ssim']:.4f} "
                f"Grad={detail['gradient']:.4f} "
                f"Align(NCC)={detail['align']:.4f} "
                f"Perc={detail['perceptual']:.4f} "
                f"[{elapsed:.1f}s]"
            )

    return meter_total.avg


def validate(model, loader, device, opts, vgg_loss=None):
    model.eval()
    meter = AverageMeter('val_loss')
    with torch.no_grad():
        for ir, vis, _ in loader:
            ir, vis = ir.to(device), vis.to(device)
            fused = model(ir, vis)
            total_loss, _ = fusion_loss(
                fused, ir, vis,
                lambda_int=opts.lambda_int,
                lambda_ssim=opts.lambda_ssim,
                lambda_grad=opts.lambda_grad,
                lambda_tv=10.0,
                offset=None,
                vgg_loss=vgg_loss,
                lambda_perceptual=opts.lambda_perceptual,
            )
            meter.update(total_loss.item())
    return meter.avg


def main(opts):
    # 设备
    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据集（兼容 data/val 和 data/test 两种目录名）
    val_split = 'val' if os.path.isdir(os.path.join(opts.data_path, 'val')) else 'test'
    train_set = FusionDataset(os.path.join(opts.data_path, 'train'),
                              patch_size=opts.patch_size, augment=True)
    val_set = FusionDataset(os.path.join(opts.data_path, val_split),
                            patch_size=opts.patch_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size,
                              shuffle=True, num_workers=opts.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers)

    # 模型
    model = LMAFusion(
        base_ch=opts.base_ch,
        d_state=opts.d_state,
        exchange_p=opts.exchange_p,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params / 1e3:.1f}K)")

    # 优化器 + 学习率调度
    optimizer = Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=opts.epoch, eta_min=opts.lr * 0.01)

    # 日志
    log_dir = os.path.join(opts.log_dir, opts.name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger = setup_logger(os.path.join(log_dir, 'train.log'))
    logger.info(f"模型参数量: {total_params:,}")

    # 断点续训
    start_epoch = 0
    best_val_loss = float('inf')
    if opts.is_resume:
        start_epoch, best_val_loss = load_model(opts.is_resume, model, optimizer, device)
        print(f"从 Epoch {start_epoch} 继续训练，历史最佳 loss={best_val_loss:.4f}")

    os.makedirs(opts.model_path, exist_ok=True)

    vgg_loss = VGGPerceptualLoss().to(device) if opts.lambda_perceptual > 0 else None

    # 训练循环
    for epoch in range(start_epoch, opts.epoch):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, writer, epoch, opts, vgg_loss=vgg_loss)
        val_loss = validate(model, val_loader, device, opts, vgg_loss=vgg_loss)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        msg = (f"Epoch [{epoch}/{opts.epoch}] "
               f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} LR={lr_now:.2e}")
        logger.info(msg)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('lr', lr_now, epoch)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(opts.model_path, 'best.pth')
            save_model(best_path, epoch, best_val_loss, model, optimizer)
            print(f"  [BEST] Saved model -> {best_path}  (val_loss={val_loss:.4f})")

        # 每 10 epoch 保存 checkpoint
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
    parser.add_argument('--lambda_grad', type=float, default=0.5, help='梯度损失权重')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='感知损失权重')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU 编号')
    parser.add_argument('--is_resume', type=str, default=None, help='断点续训权重路径')

    # 输出
    parser.add_argument('--name', type=str, default='LMAFusion', help='实验名称')
    parser.add_argument('--model_path', type=str, default='./model', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard 日志目录')
    parser.add_argument('--log_freq', type=int, default=20, help='打印频率（iteration）')

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)
