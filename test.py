"""
推理测试脚本 — LMAFusion（不依赖 cv2，使用 torchvision 读图）

用法:
  python test.py --model_path ./model/best.pth
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torchvision.io as tio

import torch
import torch.nn.functional as F

from net import LMAFusion
import torchvision.io as tio

from net import LMAFusion


def load_gray_tensor(path: str) -> torch.Tensor:
    """读取图像为灰度 tensor [1, 1, H, W]，值域 [0, 1]"""
    img = tio.read_image(str(path))         # [C, H, W], uint8
    if img.shape[0] == 3:
        # RGB → 灰度: 0.299R + 0.587G + 0.114B
        img = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
    elif img.shape[0] >= 4:
        img = img[:3]
        img = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
    else:
        img = img[0:1]
    return img.float() / 255.0


def pad_to_multiple(t: torch.Tensor, mult: int = 32):
    """右/下填充到 mult 的倍数，返回 (padded, (h_orig, w_orig))"""
    _, _, h, w = t.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph > 0 or pw > 0:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (h, w)


def infer_single(model, ir_t: torch.Tensor, vis_t: torch.Tensor, device, vis_color: np.ndarray = None):
    """
    对单对 tensor 推理，返回最终的图像 numpy 数组（若指定 vis_color，则返回彩图；否则返回灰度）
    * vis_color: [H, W, 3] uint8 (BGR)
    """
    # 确保尺寸一致（以 IR 为准）
    if ir_t.shape != vis_t.shape:
        vis_t = F.interpolate(vis_t, size=ir_t.shape[2:], mode='bilinear', align_corners=False)

    ir_t = ir_t.to(device)
    vis_t = vis_t.to(device)

    # pad 到 32 倍数
    ir_p, (h, w) = pad_to_multiple(ir_t)
    vis_p, _ = pad_to_multiple(vis_t)

    with torch.no_grad():
        out = model(ir_p, vis_p)
        # 防止越界
        out = torch.clamp(out, 0.0, 1.0)

    # 裁回原尺寸
    out = out[:, :, :h, :w]

    # 转换回 numpy [H, W] 的灰度图像
    fused_img = out.squeeze().cpu().numpy()
    fused_gray_uint8 = (fused_img * 255.0).astype(np.uint8)

    # 色彩注入后处理
    if vis_color is not None:
        # 将彩色 VIS 图 resize 到与融合图相同大小
        h, w = fused_gray_uint8.shape
        vis_bgr = cv2.resize(vis_color, (w, h), interpolation=cv2.INTER_LINEAR)
        # BGR -> YCrCb
        vis_ycrcb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)
        # 分离通道，用融合的高级纹理 Y_fusion 替换原图简单的亮度通道 Y
        y, cr, cb = cv2.split(vis_ycrcb)
        fused_ycrcb = cv2.merge([fused_gray_uint8, cr, cb])
        # YCrCb -> BGR
        final_img = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)
        return final_img

    return fused_gray_uint8


def save_gray(path: str, arr):
    """保存灰度 numpy 数组为 PNG"""
    import numpy as np
    t = torch.from_numpy(arr.astype('uint8')).unsqueeze(0)  # [1, H, W]
    tio.write_png(t, path)


def main(opts):
    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载权重 (网络存在层数或形状变更时智能加载)
    model = LMAFusion(base_ch=opts.base_ch, d_state=opts.d_state).to(device)
    try:
        ckpt = torch.load(opts.model_path, map_location=device)
        if 'model' in ckpt:
            pretrained_dict = ckpt['model']
        else:
            pretrained_dict = ckpt
            
        model_dict = model.state_dict()
        
        # 筛选出键名相同且形状完全一致的权重
        valid_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        model_dict.update(valid_dict)
        model.load_state_dict(model_dict)
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    model.eval()
    return model


def main(opts):
    # 设备
    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model = LMAFusion(base_ch=opts.base_ch, d_state=opts.d_state).to(device)
    try:
        ckpt = torch.load(opts.model_path, map_location=device)
        if 'model' in ckpt:
            pretrained_dict = ckpt['model']
        else:
            pretrained_dict = ckpt
            
        model_dict = model.state_dict()
        valid_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        model_dict.update(valid_dict)
        model.load_state_dict(model_dict)
        print(f"Model loaded: {opts.model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()
    
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_paths = sorted(list(Path(opts.ir_dir).glob('*.png'))) + sorted(list(Path(opts.ir_dir).glob('*.jpg')))
    
    total = 0
    print("Start inference...")
    for p_ir in ir_paths:
        name = p_ir.name
        # 兼容 vis 或 vi
        p_vis = Path(opts.vis_dir) / name
        p_vi = Path(opts.vis_dir).parent / 'vi' / name
        if not p_vis.exists() and p_vi.exists():
            p_vis = p_vi

        if not p_vis.exists():
            print(f"Skipping {name}: vis file not found")
            continue

        # 加载 tensor 并归一化到 [0,1]
        ir_t = load_gray_tensor(str(p_ir)).unsqueeze(0)
        vis_t = load_gray_tensor(str(p_vis)).unsqueeze(0)

        # 如果需要色彩，读取原始 BGR 图
        vis_color = None
        if opts.color:
            try:
                buf = np.fromfile(str(p_vis), dtype=np.uint8)
                vis_color = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Failed to read color image {p_vis}: {e}")
                vis_color = None

        # 推理
        fused = infer_single(model, ir_t, vis_t, device, vis_color=vis_color)

        # 保存为 png
        out_path = out_dir / name
        # 根据是否开启 color，调用不同保存方式
        if opts.color:
            cv2.imwrite(str(out_path), fused)
        else:
            save_gray(str(out_path), fused)
        total += 1
        if total % 50 == 0:
            print(f"  Processed {total} images...")

    print(f"Done! {total} fused images -> {opts.output_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./model/best.pth')
    parser.add_argument('--ir_dir', default='./data_clean/val/ir')
    parser.add_argument('--vis_dir', default='./data_clean/val/vis')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--base_ch', type=int, default=16)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--color', action='store_true', help='开启色彩融合重映射')
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)
