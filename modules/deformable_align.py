"""
模块1：可变形特征对齐模块（Deformable Feature Alignment）

用于补偿红外与可见光双光相机之间的物理视差和动态抖动偏移。

改进要点：
  1. Sobel 边缘特征 + 原始像素共 4 通道输入，跨模态匹配更鲁棒
  2. 两级粗精对齐（Coarse-to-Fine）：1/4 分辨率粗对齐 + 原始分辨率残差精对齐
  3. 可学习位移缩放因子替代硬编码 max_displacement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """提取 Sobel 梯度幅值，输入输出均为 [B, 1, H, W]"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def _make_base_grid(H: int, W: int, B: int, device: torch.device) -> torch.Tensor:
    """生成 grid_sample 所需的 [-1,1] 归一化基础网格 [B, H, W, 2]"""
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    return base.expand(B, -1, -1, -1)


def _warp(img: torch.Tensor, flow: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    根据偏移场对图像做双线性采样。

    参数:
        img   : [B, C, H, W]
        flow  : [B, 2, H, W]  Tanh 输出，值域 [-1, 1]
        scale : 标量 Tensor，控制实际像素位移 = scale * H/2
    """
    B, C, H, W = img.shape
    base_grid = _make_base_grid(H, W, B, img.device)
    offset = flow.permute(0, 2, 3, 1) * scale  # [B, H, W, 2]
    sample_grid = base_grid + offset
    return F.grid_sample(img, sample_grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


class DeformableAlignment(nn.Module):
    """
    两级粗精可变形对齐模块（Coarse-to-Fine Deformable Alignment）。

    架构:
      IR, VIS → Sobel 边缘提取 → [edge_ir, edge_vis, ir, vis] (4ch)
        → Downsample 4x → CoarsePredictor → Coarse Flow
        → Upsample + Warp IR
        → [edge_warped_ir, edge_vis, warped_ir, vis] (4ch)
        → FinePredictor → Residual Flow
        → Final Warp → aligned_ir

    参数:
        in_channels : 每路图像的通道数（灰度 = 1）
        mid_channels: 偏移预测网络的中间通道数（默认 16）
    """

    def __init__(self, in_channels: int = 1, mid_channels: int = 16):
        super().__init__()
        feat_ch = in_channels * 4  # [edge_ir, edge_vis, ir, vis]

        # ── 粗对齐网络（在 1/4 分辨率上预测全局偏移） ──────────────
        self.coarse_predictor = nn.Sequential(
            nn.Conv2d(feat_ch, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )

        # ── 精对齐网络（在原始分辨率上预测残差偏移） ────────────────
        self.fine_predictor = nn.Sequential(
            nn.Conv2d(feat_ch, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )

        # ── 可学习位移缩放因子 ───────────────────────────────────
        # coarse_scale 初始化较大（粗对齐负责大位移）
        # fine_scale 初始化较小（精对齐只修残差）
        self.coarse_scale = nn.Parameter(torch.tensor(0.08))
        self.fine_scale = nn.Parameter(torch.tensor(0.02))

        # 零初始化输出层，训练初期不做偏移
        for predictor in [self.coarse_predictor, self.fine_predictor]:
            last_conv = predictor[-2]  # Tanh 前面的 Conv2d
            nn.init.constant_(last_conv.weight, 0)
            nn.init.constant_(last_conv.bias, 0)

    def _build_input(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """构造 4 通道输入: [edge_ir, edge_vis, ir, vis]"""
        edge_ir = _sobel_edges(ir)
        edge_vis = _sobel_edges(vis)
        return torch.cat([edge_ir, edge_vis, ir, vis], dim=1)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor):
        """
        参数:
            ir  (Tensor): 红外图像 [B, 1, H, W]，值域 [0, 1]
            vis (Tensor): 可见光图像 [B, 1, H, W]，值域 [0, 1]

        返回:
            aligned_ir (Tensor): 对齐后红外 [B, 1, H, W]
            vis        (Tensor): 透传可见光
            total_flow (Tensor): 总偏移场 [B, 2, H, W]（用于 TV 正则）
        """
        B, C, H, W = ir.shape

        # ── Stage 1: 粗对齐（1/4 分辨率） ─────────────────────────
        ir_down = F.interpolate(ir, scale_factor=0.25, mode='bilinear', align_corners=False)
        vis_down = F.interpolate(vis, scale_factor=0.25, mode='bilinear', align_corners=False)

        coarse_input = self._build_input(ir_down, vis_down)
        coarse_flow_small = self.coarse_predictor(coarse_input)  # [B, 2, H/4, W/4]

        # 上采样到原始分辨率
        coarse_flow = F.interpolate(coarse_flow_small, size=(H, W),
                                    mode='bilinear', align_corners=False)

        # 粗对齐 warp
        coarse_scale = self.coarse_scale.clamp(0.01, 0.3)
        ir_coarse_aligned = _warp(ir, coarse_flow, coarse_scale)

        # ── Stage 2: 精对齐（原始分辨率，预测残差） ────────────────
        fine_input = self._build_input(ir_coarse_aligned, vis)
        fine_flow = self.fine_predictor(fine_input)  # [B, 2, H, W]

        # 精对齐 warp
        fine_scale = self.fine_scale.clamp(0.005, 0.1)
        aligned_ir = _warp(ir_coarse_aligned, fine_flow, fine_scale)

        # 总偏移场（供 TV 正则化使用）
        total_flow = coarse_flow * coarse_scale + fine_flow * fine_scale

        return aligned_ir, vis, total_flow
