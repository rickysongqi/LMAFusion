"""
模块1：可变形特征对齐模块（Deformable Feature Alignment）

用于补偿红外与可见光双光相机之间的物理视差和动态抖动偏移。
使用 torchvision 的 deform_conv2d 或降级到 grid_sample 实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAlignment(nn.Module):
    """
    轻量可变形对齐模块。

    将可见光图像作为参考基准，预测红外图像的偏移场，
    并通过可变形卷积（或双线性网格采样）将红外特征在空间上对齐。

    参数:
        in_channels: 输入通道数（灰度图像默认为 1）
        mid_channels: 偏移场预测网络的中间通道数（轻量化，默认 16）
    """

    def __init__(self, in_channels: int = 1, mid_channels: int = 16):
        super().__init__()

        # 偏移场预测网络：采用空洞卷积扩大感受野，用于涵盖大范围抖动
        # 输入: [B, 2*in_channels, H, W]  (IR + VIS 拼接)
        # 输出: [B, 2, H, W]              (x, y 方向的像素级偏移)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),  # 将偏移限制在 [-1, 1] 范围（相对位移）
        )

        # 初始化：让偏移场初始值接近 0（即初始不做任何偏移）
        nn.init.constant_(self.offset_predictor[-2].weight, 0)
        nn.init.constant_(self.offset_predictor[-2].bias, 0)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor):
        """
        前向传播。

        参数:
            ir  (Tensor): 红外图像特征 [B, C, H, W]
            vis (Tensor): 可见光图像特征 [B, C, H, W]（作为对齐参考）

        返回:
            aligned_ir (Tensor): 空间对齐后的红外特征 [B, C, H, W]
            vis        (Tensor): 不变的可见光特征（透传）
            offset     (Tensor): 偏移场网格 [B, 2, H, W]
        """
        B, C, H, W = ir.shape

        # 1. 拼接两路特征，预测偏移场 [B, 2, H, W]
        concat_feat = torch.cat([ir, vis], dim=1)
        offset = self.offset_predictor(concat_feat)  # [-1, 1] 范围的相对偏移

        # 2. 将偏移场转换为 grid_sample 所需的绝对采样坐标
        # grid_sample 要求网格坐标在 [-1, 1] 范围内 (归一化坐标系)
        # 生成基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=ir.device),
            torch.linspace(-1, 1, W, device=ir.device),
            indexing='ij'
        )
        # 基础网格: [1, H, W, 2]  (x, y)
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        base_grid = base_grid.expand(B, -1, -1, -1)                     # [B, H, W, 2]

        # 偏移量: [B, 2, H, W] -> [B, H, W, 2]
        offset_perm = offset.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # 采样网格 = 基础网格 + 偏移量（偏移量已在 [-1,1] 内，小偏移对应小像素位移）
        # 实际像素最大位移变为 max_displacement * H/2 像素
        max_displacement = 0.1  # 恢复到较小的安全约束，防止在已对齐数据集上产生鬼影漂移
        sample_grid = base_grid + offset_perm * max_displacement

        # 3. 使用双线性插值对红外图像进行空间变形采样
        aligned_ir = F.grid_sample(
            ir,
            sample_grid,
            mode='bilinear',
            padding_mode='border',  # 边界外的像素用边界值填充，避免黑边
            align_corners=True
        )

        return aligned_ir, vis, offset


class DeformableAlignmentWithFlow(nn.Module):
    """
    更强版本：显式光流网络预测（备选方案，参数量略多）
    使用两阶段级联偏移预测，提高对大位移的鲁棒性。
    """

    def __init__(self, in_channels: int = 1, flow_channels: int = 32):
        super().__init__()

        # 粗粒度流场预测（整体平移估计）
        self.coarse_flow = nn.Sequential(
            nn.Conv2d(in_channels * 2, flow_channels, 5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_channels, 2, 3, padding=1, bias=True),
            nn.Tanh()
        )
        # 细粒度残差流场预测（局部精调）
        self.fine_flow = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 2, flow_channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_channels // 2, 2, 3, padding=1, bias=True),
            nn.Tanh()
        )

    def warp(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """根据光流场对图像进行双线性采样"""
        B, C, H, W = img.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        sample_grid = base + flow.permute(0, 2, 3, 1) * 0.1
        return F.grid_sample(img, sample_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor):
        concat = torch.cat([ir, vis], dim=1)
        coarse = self.coarse_flow(concat)
        warped_ir_coarse = self.warp(ir, coarse)
        concat_refined = torch.cat([warped_ir_coarse, vis, coarse], dim=1)
        fine = self.fine_flow(concat_refined)
        aligned_ir = self.warp(warped_ir_coarse, fine)
        return aligned_ir, vis
