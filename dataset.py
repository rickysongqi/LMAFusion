"""
数据集加载模块

支持无人机红外/可见光双光数据集（同文件名 PNG 格式配对）。
数据集目录结构：
    data/
      train/
        ir/   *.png  (红外，单通道灰度或彩色→灰度)
        vis/  *.png  (可见光，单通道灰度)
      val/
        ir/   *.png
        vis/  *.png

使用方法：
    # 复制数据集图像到 data/train/ir 和 data/train/vis
    dataset = FusionDataset('./data/train', patch_size=128, augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FusionDataset(Dataset):
    """
    红外/可见光图像融合数据集。

    参数:
        data_dir   (str ) : 数据根目录（含 ir/ 和 vis/ 子目录）
        patch_size (int ) : 随机裁剪 patch 大小（None = 不裁剪，用全图）
        augment    (bool) : 是否进行数据增强（翻转/亮度抖动）
        i2_dir     (str ) : 自引导图像 I2 目录（AGM 自适应引导训练用）
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 128,
        augment: bool = True,
        vis_crop: tuple = None,
        i2_dir: str = None,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.vis_crop = vis_crop
        self.i2_dir = Path(i2_dir) if i2_dir else None

        ir_dir = Path(data_dir) / 'ir'
        vis_dir = Path(data_dir) / 'vis'
        if not vis_dir.exists():
            vis_dir = Path(data_dir) / 'vi'
        if not vis_dir.exists():
            raise FileNotFoundError(f"在 {data_dir} 下未找到 vis/ 或 vi/ 子目录")

        ir_names = {f.name for f in ir_dir.glob('*.png')}
        ir_names |= {f.name for f in ir_dir.glob('*.jpg')}
        vis_names = {f.name for f in vis_dir.glob('*.png')}
        vis_names |= {f.name for f in vis_dir.glob('*.jpg')}

        common_names = sorted(ir_names & vis_names)
        if len(common_names) == 0:
            raise FileNotFoundError(
                f"在 {data_dir} 下未找到 IR/VIS 配对图像。\n"
                f"请确保 ir/ 和 vis/ 目录中有同名的 PNG/JPG 文件。"
            )

        self.ir_paths = [str(ir_dir / name) for name in common_names]
        self.vis_paths = [str(vis_dir / name) for name in common_names]
        self.names = common_names
        print(f"数据集 [{data_dir}] 加载成功：{len(self.ir_paths)} 对图像")

    def __len__(self) -> int:
        return len(self.ir_paths)

    def _load_gray(self, path: str) -> np.ndarray:
        """加载图像并转换为灰度，归一化至 [0, 1]
        注：使用 np.fromfile + imdecode 以支持含中文字符的路径（cv2.imread 不支持）
        """
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise IOError(f"无法读取图像: {path}")
        return img.astype(np.float32) / 255.0

    def _random_crop(self, *images: np.ndarray) -> list:
        """对一组图像做相同的随机裁剪"""
        h, w = images[0].shape[:2]
        max_y = max(0, h - self.patch_size)
        max_x = max(0, w - self.patch_size)
        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)
        return [img[y0:y0 + self.patch_size, x0:x0 + self.patch_size] for img in images]

    def _augment(self, ir: np.ndarray, vis: np.ndarray) -> tuple:
        """数据增强：随机水平/垂直翻转，可见光亮度抖动"""
        # 随机水平翻转
        if random.random() > 0.5:
            ir = np.fliplr(ir)
            vis = np.fliplr(vis)
        # 随机垂直翻转
        if random.random() > 0.3:
            ir = np.flipud(ir)
            vis = np.flipud(vis)
        # 可见光亮度随机扰动（+- 10%）
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            vis = np.clip(vis * factor, 0.0, 1.0)
        return ir, vis

    def _load_i2(self, name: str, target_h: int, target_w: int) -> np.ndarray:
        """加载 I2 引导图像；不存在则返回全白图（值 1.0）"""
        if self.i2_dir is None:
            return np.ones((target_h, target_w), dtype=np.float32)
        # I2 统一存为 .jpg
        stem = Path(name).stem
        i2_path = self.i2_dir / f"{stem}.jpg"
        if not i2_path.exists():
            i2_path = self.i2_dir / name
        if i2_path.exists():
            try:
                return self._load_gray(str(i2_path))
            except IOError:
                pass
        return np.ones((target_h, target_w), dtype=np.float32)

    def __getitem__(self, idx: int) -> tuple:
        ir = self._load_gray(self.ir_paths[idx])
        vis = self._load_gray(self.vis_paths[idx])
        name = os.path.basename(self.ir_paths[idx])

        if self.vis_crop is not None:
            x1, y1, x2, y2 = self.vis_crop
            vis = vis[y1:y2, x1:x2]
        ir_h, ir_w = ir.shape[:2]
        vis_h, vis_w = vis.shape[:2]
        if (vis_h, vis_w) != (ir_h, ir_w):
            vis = cv2.resize(vis, (ir_w, ir_h), interpolation=cv2.INTER_LINEAR)

        i2 = self._load_i2(name, ir_h, ir_w)
        if i2.shape[:2] != (ir_h, ir_w):
            i2 = cv2.resize(i2, (ir_w, ir_h), interpolation=cv2.INTER_LINEAR)

        if self.patch_size is not None:
            h, w = ir.shape[:2]
            if h >= self.patch_size and w >= self.patch_size:
                ir, vis, i2 = self._random_crop(ir, vis, i2)
            else:
                ir = cv2.resize(ir, (self.patch_size, self.patch_size))
                vis = cv2.resize(vis, (self.patch_size, self.patch_size))
                i2 = cv2.resize(i2, (self.patch_size, self.patch_size))

        if self.augment:
            ir, vis = self._augment(ir, vis)

        ir_t = torch.from_numpy(np.ascontiguousarray(ir)).unsqueeze(0)
        vis_t = torch.from_numpy(np.ascontiguousarray(vis)).unsqueeze(0)
        i2_t = torch.from_numpy(np.ascontiguousarray(i2)).unsqueeze(0)

        return ir_t, vis_t, i2_t, name


class DroneDatasetPreparer:
    """
    无人机数据集预处理工具。

    将原始 YOLO 格式数据集（train_by_yolo_IR / train_by_yolo_RGB）
    转换为 LMAFusion 所需的 data/train/ir 和 data/train/vis 格式。

    同名文件配对：
        train_by_yolo_IR/images/train/*.png  →  data/train/ir/
        train_by_yolo_RGB/images/train/*.png →  data/train/vis/
    """

    @staticmethod
    def prepare(
        ir_src: str,
        vis_src: str,
        output_dir: str,
        max_sample: int = None,
        val_ratio: float = 0.1,
    ):
        """
        准备数据集，将图像软链接（Windows：复制）到目标目录。

        参数:
            ir_src     : 红外图像源目录（如 .../train_by_yolo_IR/images/train）
            vis_src    : 可见光图像源目录（如 .../train_by_yolo_RGB/images/train）
            output_dir : 输出根目录（如 ./data）
            max_sample : 采样上限（None = 全部）
            val_ratio  : 验证集比例
        """
        import shutil
        ir_src = Path(ir_src)
        vis_src = Path(vis_src)
        output_dir = Path(output_dir)

        # 找公共文件
        ir_files = {f.name: f for f in ir_src.glob('*.png')}
        ir_files.update({f.name: f for f in ir_src.glob('*.jpg')})
        vis_files = {f.name: f for f in vis_src.glob('*.png')}
        vis_files.update({f.name: f for f in vis_src.glob('*.jpg')})

        common = sorted(ir_files.keys() & vis_files.keys())
        if max_sample:
            common = common[:max_sample]

        # 划分训练/验证集
        n_val = max(1, int(len(common) * val_ratio))
        val_names = set(random.sample(common, n_val))
        train_names = [n for n in common if n not in val_names]

        splits = {'train': train_names, 'val': list(val_names)}
        for split, names in splits.items():
            for modal in ['ir', 'vis']:
                (output_dir / split / modal).mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(str(ir_files[name]), str(output_dir / split / 'ir' / name))
                shutil.copy2(str(vis_files[name]), str(output_dir / split / 'vis' / name))
            print(f"[{split}] {len(names)} 对图像 → {output_dir / split}")

        print("数据集准备完成！")


if __name__ == '__main__':
    # 快速测试
    import sys
    ds = FusionDataset('./data/train', patch_size=128, augment=True)
    ir, vis, name = ds[0]
    print(f"样例: {name}, IR shape: {ir.shape}, VIS shape: {vis.shape}")
    print(f"IR 值域: [{ir.min():.3f}, {ir.max():.3f}]")
    print(f"VIS 值域: [{vis.min():.3f}, {vis.max():.3f}]")
