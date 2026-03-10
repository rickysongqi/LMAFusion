"""
OSD 区域裁剪预处理脚本

反无人机数据集的图像中包含 HUD 界面叠加元素（OSD），
主要包括：
  - 顶部：相机类型文字 + 时间戳（约 70px）
  - 底部：目标类别标签    （约 40px）

这些 UI 元素会干扰对齐网络学习真正的目标位移，
本脚本将其裁剪后保存到新目录，再用于训练。

用法：
    python crop_osd.py --src_dir ./data --dst_dir ./data_clean
    python crop_osd.py --src_dir ./data --dst_dir ./data_clean --top 70 --bottom 40
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def crop_image(img: np.ndarray,
               top_ratio: float = 0.18,
               bottom_ratio: float = 0.10) -> np.ndarray:
    """按比例裁掉顶部和底部 OSD 区域（自动适配不同分辨率的 IR 和 VIS 相机）"""
    h = img.shape[0]
    top_px = int(h * top_ratio)
    bottom_px = int(h * bottom_ratio)
    return img[top_px: h - bottom_px, :]


def process_split(src_root: Path, dst_root: Path, split: str,
                  top_ratio: float, bottom_ratio: float, dry_run: bool = False):
    """处理 train 或 val 下的 ir/ 和 vis/ 两个子目录"""
    for modal in ['ir', 'vis']:
        src_dir = src_root / split / modal
        dst_dir = dst_root / split / modal

        if not src_dir.exists():
            print(f"  [SKIP] 目录不存在: {src_dir}")
            continue

        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)

        imgs = list(src_dir.glob('*.png')) + list(src_dir.glob('*.jpg'))
        print(f"  [{split}/{modal}] 处理 {len(imgs)} 张图像 → {dst_dir}")

        for img_path in imgs:
            if dry_run:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    警告：无法读取 {img_path.name}")
                continue
            cropped = crop_image(img, top_ratio=top_ratio, bottom_ratio=bottom_ratio)
            out_path = dst_dir / img_path.name
            cv2.imwrite(str(out_path), cropped)


def main():
    parser = argparse.ArgumentParser(description="裁剪数据集 OSD 区域")
    parser.add_argument('--src_dir', default='./data',
                        help='原始数据集根目录（含 train/ 和 val/）')
    parser.add_argument('--dst_dir', default='./data_clean',
                        help='清洁版数据集输出目录')
    parser.add_argument('--top_ratio', type=float, default=0.18,
                        help='顶部裁剪比例（默认 0.18，按图高的 18%%，自动适配不同分辨率）')
    parser.add_argument('--bottom_ratio', type=float, default=0.10,
                        help='底部裁剪比例（默认 0.10，按图高的 10%%）')
    parser.add_argument('--dry_run', action='store_true',
                        help='只统计，不实际裁剪')
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)

    print(f"=== OSD 裁剪预处理 ===")
    print(f"  源目录     : {src_root}")
    print(f"  目标目录   : {dst_root}")
    print(f"  顶部裁剪   : {args.top_ratio*100:.0f}% of H")
    print(f"  底部裁剪   : {args.bottom_ratio*100:.0f}% of H")
    print(f"  空跑模式   : {'是' if args.dry_run else '否'}")
    print()

    for split in ['train', 'val']:
        process_split(src_root, dst_root, split,
                      top_ratio=args.top_ratio, bottom_ratio=args.bottom_ratio,
                      dry_run=args.dry_run)

    print()
    print("裁剪完成！")
    print(f"下一步：用 --data_dir {args.dst_dir} 参数重新训练：")
    print(f"  python train.py --data_dir {args.dst_dir}")


if __name__ == '__main__':
    main()
