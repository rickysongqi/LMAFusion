# 快速数据集验证脚本 - 在 AGMFusion conda 环境中运行
# 用法: python test_dataset.py
from dataset import FusionDataset
import torch

print("测试 vis_crop 参数 ...")
VIS_CROP = (654, 0, 1325, 387)
ds = FusionDataset('./data_clean/train', patch_size=128, augment=False, vis_crop=VIS_CROP)
ir, vis, name = ds[0]
print(f"  样本: {name}")
print(f"  IR  shape: {ir.shape}, range: [{ir.min():.3f}, {ir.max():.3f}]")
print(f"  VIS shape: {vis.shape}, range: [{vis.min():.3f}, {vis.max():.3f}]")
assert ir.shape == vis.shape, f"形状不一致: IR={ir.shape}, VIS={vis.shape}"
assert ir.shape == (1, 128, 128), f"期望 (1,128,128), 实际 {ir.shape}"
print("数据集测试通过!")
print(f"\nNCC 损失函数测试 ...")
from losses import ncc_loss
a = torch.rand(2, 1, 128, 128)
b = torch.rand(2, 1, 128, 128)
loss = ncc_loss(a, b)
loss_self = ncc_loss(a, a)
print(f"  ncc_loss(random, random) = {loss.item():.4f}")
print(f"  ncc_loss(x, x)          = {loss_self.item():.4f} (应接近 0)")
assert loss_self.item() < 0.05, "自相关 NCC 应接近 0"
print("NCC 损失测试通过!")
