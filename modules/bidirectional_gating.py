"""
模块4：双向深度特征显式门控融合（Bidirectional Deep State Fusion, DSSF）

利用红外的热力响应和可见光的边缘结构互为约束，
通过正反向两路 Sigmoid 门控（哈达玛积）联合抑制：
- 正向：用红外热图抑制 VIS 中无热量的伪目标（飞鸟、塑料袋等）
- 反向：用 VIS 边缘结构抑制 IR 的环境热量泛光噪声

最终通过解码器重建高质量融合图像。
"""

import torch
import torch.nn as nn


class BidirectionalGating(nn.Module):
    """
    双向特征显式门控模块（核心融合模块）。

    参数:
        channels (int): 深层特征的通道数
    """

    def __init__(self, channels: int):
        super().__init__()

        # 正向门控生成器：IR 特征 → 控制 VIS 的权重图
        # 抑制 VIS 中没有热能响应的区域（伪目标）
        self.gate_ir2vis = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 反向门控生成器：VIS 特征 → 控制 IR 的权重图
        # 抑制 IR 中没有清晰边缘能量的背景热辐射噪声
        self.gate_vis2ir = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor) -> torch.Tensor:
        """
        双向门控融合前向传播。

        参数:
            feat_ir  (Tensor): 红外深层特征 [B, C, H, W]
            feat_vis (Tensor): 可见光深层特征 [B, C, H, W]

        返回:
            fused (Tensor): 融合后特征 [B, C, H, W]
        """
        # 拼接两路深层特征
        concat = torch.cat([feat_ir, feat_vis], dim=1)  # [B, 2C, H, W]

        # 正向门：IR 激发，生成对 VIS 特征的抑制/增强权重
        gate_for_vis = self.gate_ir2vis(concat)    # [B, C, H, W]，值域 (0,1)

        # 反向门：VIS 结构引导，生成对 IR 特征的抑制/增强权重
        gate_for_ir = self.gate_vis2ir(concat)     # [B, C, H, W]，值域 (0,1)

        # 哈达玛积（逐元素乘）：门控特征加权融合
        # 正向：VIS 特征被 IR 热力图"筛选"——无热量的冗余区域被抑制
        # 反向：IR 特征被 VIS 边缘结构"约束"——泛光区域被压制
        fused = feat_vis * gate_for_vis + feat_ir * gate_for_ir

        return fused


class FusionDecoder(nn.Module):
    """
    融合解码器：将门控融合后的深层特征重建为单通道融合图像。

    参数:
        in_channels  (int): 输入特征通道数（等于深层特征通道数）
        mid_channels (int): 解码中间层通道数
    """

    def __init__(self, in_channels: int, mid_channels: int = 16):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x   (Tensor): [B, in_channels, H, W]

        返回:
            out (Tensor): [B, 1, H, W]，值域 [0, 1]
        """
        return self.decoder(x)
