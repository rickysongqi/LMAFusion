"""
图像质量指标评估脚本（不依赖 cv2 / skimage，使用纯 PyTorch + torchvision）

指标:
  PSNR  - 峰值信噪比（越高越好，对比VIS）
  SSIM  - 结构相似度（越高越好，对比VIS）
  EN    - 信息熵（越高越好，衡量信息量）
  MI_ir - 融合图与IR的互信息（越高越好）
  MI_vis- 融合图与VIS的互信息（越高越好）
  SF    - 空间频率（越高越好，细节丰富度）
  AG    - 平均梯度（越高越好，图像清晰度）

用法:
  python evaluate.py --fused_dir ./results --ir_dir ./data/val/ir --vis_dir ./data/val/vis
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tio


# ── 图像读取 ──────────────────────────────────────────────────
def read_gray(path: str) -> np.ndarray:
    """读取灰度图像，返回 uint8 numpy [H, W]"""
    img = tio.read_image(str(path))   # [C, H, W] uint8
    if img.shape[0] >= 3:
        gray = (0.299 * img[0].float() + 0.587 * img[1].float() + 0.114 * img[2].float())
        return gray.numpy().astype(np.uint8)
    else:
        return img[0].numpy()


# ── 指标函数 ──────────────────────────────────────────────────
def compute_psnr(img: np.ndarray, ref: np.ndarray) -> float:
    img_f = img.astype(np.float64)
    ref_f = ref.astype(np.float64)
    mse = np.mean((img_f - ref_f) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img: np.ndarray, ref: np.ndarray,
                 win: int = 11, sigma: float = 1.5) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img_t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    ref_t = torch.from_numpy(ref.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    coords = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    pad = win // 2

    mu1 = F.conv2d(img_t, kernel, padding=pad)
    mu2 = F.conv2d(ref_t, kernel, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = F.conv2d(img_t ** 2, kernel, padding=pad) - mu1_sq
    s2 = F.conv2d(ref_t ** 2, kernel, padding=pad) - mu2_sq
    s12 = F.conv2d(img_t * ref_t, kernel, padding=pad) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float((num / den).mean().item())


def compute_entropy(img: np.ndarray) -> float:
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_mi(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.histogram(a.ravel(), 256, (0, 256))[0] / a.size + 1e-12
    pb = np.histogram(b.ravel(), 256, (0, 256))[0] / b.size + 1e-12
    pab, _, _ = np.histogram2d(a.ravel(), b.ravel(), 256, [[0, 256], [0, 256]])
    pab = pab / (pab.sum() + 1e-12) + 1e-12
    return float(np.sum(pab * np.log2(pab / (pa[:, None] * pb[None, :]))))


def compute_sf(img: np.ndarray) -> float:
    rf = np.sqrt(np.mean((img[:, 1:].astype(float) - img[:, :-1].astype(float)) ** 2))
    cf = np.sqrt(np.mean((img[1:, :].astype(float) - img[:-1, :].astype(float)) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def compute_ag(img: np.ndarray) -> float:
    """平均梯度（Sobel 近似）"""
    img_f = img.astype(np.float64)
    gx = img_f[:, 1:] - img_f[:, :-1]
    gy = img_f[1:, :] - img_f[:-1, :]
    # 对齐尺寸
    gx = gx[:gy.shape[0], :gy.shape[1]]
    gy = gy[:gx.shape[0], :gx.shape[1]]
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2)))


# ── 主函数 ──────────────────────────────────────────────────
def evaluate(fused_dir, ir_dir, vis_dir):
    fused_dir = Path(fused_dir)
    ir_dir = Path(ir_dir)
    vis_dir = Path(vis_dir)

    files = sorted(list(fused_dir.glob('*.png')) + list(fused_dir.glob('*.jpg')))
    results = []

    for fp in files:
        ip = ir_dir / fp.name
        vp = vis_dir / fp.name
        if not (ip.exists() and vp.exists()):
            continue

        fused = read_gray(fp)
        ir = read_gray(ip)
        vis = read_gray(vp)

        # 统一尺寸（以融合图为准）
        h, w = fused.shape
        if ir.shape != (h, w):
            ir_t = torch.from_numpy(ir).float().unsqueeze(0).unsqueeze(0)
            ir = F.interpolate(ir_t, size=(h, w), mode='bilinear',
                               align_corners=False).squeeze().numpy().astype(np.uint8)
        if vis.shape != (h, w):
            vis_t = torch.from_numpy(vis).float().unsqueeze(0).unsqueeze(0)
            vis = F.interpolate(vis_t, size=(h, w), mode='bilinear',
                                align_corners=False).squeeze().numpy().astype(np.uint8)

        results.append({
            'name': fp.name,
            'PSNR_vis': compute_psnr(fused, vis),
            'PSNR_ir':  compute_psnr(fused, ir),
            'SSIM_vis': compute_ssim(fused, vis),
            'SSIM_ir':  compute_ssim(fused, ir),
            'EN':       compute_entropy(fused),
            'MI_ir':    compute_mi(fused, ir),
            'MI_vis':   compute_mi(fused, vis),
            'SF':       compute_sf(fused),
            'AG':       compute_ag(fused),
        })

    if not results:
        print("No matched image pairs found!")
        return

    keys = [k for k in results[0] if k != 'name']
    means = {k: float(np.mean([r[k] for r in results])) for k in keys}

    # 打印结果表
    print(f"\n{'='*60}")
    print(f"  LMAFusion Evaluation Results  ({len(results)} images)")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Mean':>10}")
    print(f"  {'-'*25}")
    for k, v in means.items():
        print(f"  {k:<12} {v:>10.4f}")
    print(f"{'='*60}\n")

    # 保存 CSV
    csv_path = str(fused_dir / 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name'] + keys)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({'name': 'MEAN', **means})
    print(f"Saved: {csv_path}")
    return means


def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--fused_dir', default='./results')
    p.add_argument('--ir_dir', default='./data/val/ir')
    p.add_argument('--vis_dir', default='./data/val/vis')
    return p.parse_args()


if __name__ == '__main__':
    opts = parse_opt()
    evaluate(opts.fused_dir, opts.ir_dir, opts.vis_dir)
