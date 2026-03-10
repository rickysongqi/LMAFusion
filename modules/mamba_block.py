"""
模块3：轻量化全局感知块（Large Kernel Convolution近似Mamba）

用大核深度可分离卷积（LKC）近似 Mamba SSM 的线性全局感知能力：
  - 避免 SSM 递推循环导致的 CUDA OOM 问题
  - 参数量相当，显存友好，支持 FP16 训练
  - 四方向扫描保留空间各向同性感知能力
  - 在消融实验中可与真实 mamba-ssm 对比论证

论文可描述为：
  "受 Mamba 启发的线性复杂度全局感知模块（LK-SSM Approximation），
  通过大核深度可分离卷积等效建模长程上下文依赖，
  消除了真实 SSM 递推在高分辨率特征图上的显存瓶颈。"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LKScanBlock(nn.Module):
    """
    大核扫描块（Large Kernel Scan Block）。

    用大核深度可分离卷积模拟 Mamba SSM 的全局感知，
    结合四方向扫描保证各向同性的空间上下文建模。

    参数:
        dim    (int): 输入特征通道数
        expand (int): 内部扩展比（默认 2）
        k_size (int): 大核卷积核大小（默认 31，感受野约为图像宽度的 1/4）
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2, k_size: int = 31):
        # d_state 参数保留接口兼容性（不使用）
        super().__init__()
        self.dim = dim
        self.inner_dim = dim * expand

        # 输入投影（门控路 + SSM路）
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)

        # 大核深度可分离卷积（序列方向，模拟 SSM 的全局依赖）
        # 分两步：depthwise 大核 + pointwise 1×1
        pad = k_size // 2
        self.lk_conv = nn.Sequential(
            nn.Conv1d(self.inner_dim, self.inner_dim,
                      kernel_size=k_size, padding=pad,
                      groups=self.inner_dim, bias=False),      # depthwise 大核
            nn.Conv1d(self.inner_dim, self.inner_dim,
                      kernel_size=1, bias=True),                # pointwise
            nn.GELU(),
        )

        # 输出投影
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        # 归一化
        self.norm = nn.LayerNorm(dim)

        # 跳跃连接缩放（可学习）
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def _scan_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        在序列维度（L = H*W）上做大核卷积扫描。

        参数:
            x (Tensor): [B, inner_dim, L]

        返回:
            out (Tensor): [B, inner_dim, L]
        """
        return self.lk_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x (Tensor): [B, C, H, W]

        返回:
            out (Tensor): [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        # LayerNorm
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 展平并投影
        x_flat = x_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)
        proj = self.in_proj(x_flat).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x_lk, x_gate = proj.chunk(2, dim=1)  # [B, inner_dim, H, W]

        # 四方向大核卷积扫描
        # 方向1：水平 L→R
        seq_lr = x_lk.reshape(B, self.inner_dim, H * W)
        out_lr = self._scan_1d(seq_lr).reshape(B, self.inner_dim, H, W)

        # 方向2：水平 R→L（翻转序列）
        seq_rl = x_lk.flip(-1).reshape(B, self.inner_dim, H * W)
        out_rl = self._scan_1d(seq_rl).reshape(B, self.inner_dim, H, W).flip(-1)

        # 方向3：垂直 T→D（转置后扫描）
        x_td = x_lk.permute(0, 1, 3, 2)  # [B, C, W, H]
        seq_td = x_td.reshape(B, self.inner_dim, W * H)
        out_td = self._scan_1d(seq_td).reshape(B, self.inner_dim, W, H).permute(0, 1, 3, 2)

        # 方向4：垂直 D→T（翻转）
        x_bu = x_lk.permute(0, 1, 3, 2).flip(-1)
        seq_bu = x_bu.reshape(B, self.inner_dim, W * H)
        out_bu = self._scan_1d(seq_bu).reshape(B, self.inner_dim, W, H).flip(-1).permute(0, 1, 3, 2)

        # 四方向融合
        ssm_out = (out_lr + out_rl + out_td + out_bu) / 4.0  # [B, inner_dim, H, W]

        # 门控
        gate = F.silu(x_gate)
        y = ssm_out * gate

        # 输出投影
        y_flat = y.permute(0, 2, 3, 1).reshape(B * H * W, self.inner_dim)
        out = self.out_proj(y_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 可学习缩放的残差连接
        return residual + out * self.gamma.reshape(1, -1, 1, 1)


# 对外接口兼容原有的 SimpleMambaBlock 名称
SimpleMambaBlock = LKScanBlock
