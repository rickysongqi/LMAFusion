"""
数据集准备脚本

将无人机双光数据集从 YOLO 格式转换到 LMAFusion 格式：
  输入:  .../train_by_yolo_IR/images/train/*.png
         .../train_by_yolo_RGB/images/train/*.png
  输出:  ./data/train/ir/*.png
         ./data/train/vis/*.png
         ./data/val/ir/*.png
         ./data/val/vis/*.png

用法:
  python prepare_data.py
  python prepare_data.py --max_sample 1000  # 只取前 1000 对（快速测试）
  python prepare_data.py --val_ratio 0.1    # 10% 作为验证集
"""

import argparse
import os
import random
import shutil
from pathlib import Path


DATASET_BASE = r"C:\Users\Ricky-Li\Desktop\毕业设计\可见光（RGB）和红外（IR）的无人机检测数据集 均有5104张训练集"

DEFAULT_IR_SRC = os.path.join(DATASET_BASE, "train_by_yolo_IR", "images", "train")
DEFAULT_VIS_SRC = os.path.join(DATASET_BASE, "train_by_yolo_RGB", "images", "train")


def prepare_dataset(ir_src: str, vis_src: str, output_dir: str,
                    max_sample: int = None, val_ratio: float = 0.1,
                    seed: int = 42):
    random.seed(seed)
    ir_src = Path(ir_src)
    vis_src = Path(vis_src)
    output_dir = Path(output_dir)

    print(f"红外源目录: {ir_src}")
    print(f"可见光源目录: {vis_src}")

    # 找同名文件（配对）
    ir_files = {f.name: f for f in ir_src.iterdir() if f.suffix in ('.png', '.jpg')}
    vis_files = {f.name: f for f in vis_src.iterdir() if f.suffix in ('.png', '.jpg')}

    common = sorted(ir_files.keys() & vis_files.keys())
    print(f"发现配对图像: {len(common)} 对")

    if max_sample and max_sample < len(common):
        common = sorted(random.sample(common, max_sample))
        print(f"采样使用: {max_sample} 对")

    # 划分训练/验证集
    n_val = max(1, int(len(common) * val_ratio))
    val_names = set(random.sample(common, n_val))
    train_names = [n for n in common if n not in val_names]

    splits = {'train': train_names, 'val': sorted(val_names)}
    for split, names in splits.items():
        ir_dst = output_dir / split / 'ir'
        vis_dst = output_dir / split / 'vis'
        ir_dst.mkdir(parents=True, exist_ok=True)
        vis_dst.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(names):
            shutil.copy2(str(ir_files[name]), str(ir_dst / name))
            shutil.copy2(str(vis_files[name]), str(vis_dst / name))
            if (i + 1) % 200 == 0:
                print(f"  [{split}] 已复制 {i+1}/{len(names)}...")

        print(f"✓ [{split:5s}] {len(names):5d} 对 → {output_dir / split}")

    print("\n数据集准备完成！")
    print(f"  训练集: {len(train_names)} 对")
    print(f"  验证集: {len(list(val_names))} 对")


def parse_opt():
    parser = argparse.ArgumentParser(description='LMAFusion 数据集准备')
    parser.add_argument('--ir_src', type=str, default=DEFAULT_IR_SRC)
    parser.add_argument('--vis_src', type=str, default=DEFAULT_VIS_SRC)
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--max_sample', type=int, default=None,
                        help='最大采样数（None=全部）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_opt()
    prepare_dataset(
        ir_src=opts.ir_src,
        vis_src=opts.vis_src,
        output_dir=opts.output_dir,
        max_sample=opts.max_sample,
        val_ratio=opts.val_ratio,
    )
