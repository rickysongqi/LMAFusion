"""
Qabf — 基于梯度的融合质量指标 (Xydeas & Petrovic, 2000)
精简自 IVIF_ZOO-main/Metric/Qabf.py，仅依赖 numpy + scipy。
"""
import math
import numpy as np
from scipy.signal import convolve2d


def _per_extn(x, wsize):
    hw = (wsize - 1) // 2
    p, q = x.shape
    out = np.zeros((p + wsize - 1, q + wsize - 1))
    out[hw:p + hw, hw:q + hw] = x
    if wsize - 1 == hw + 1:
        out[:hw, :] = out[2, :].reshape(1, -1)
        out[p + hw:p + wsize - 1, :] = out[-3, :].reshape(1, -1)
    out[:, :hw] = out[:, 2].reshape(-1, 1)
    out[:, q + hw:q + wsize - 1] = out[:, -3].reshape(-1, 1)
    return out


def compute_qabf(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> float:
    """
    计算 Qabf（越高越好，取值 0~1）。
    输入均为 2-D float64 / uint8 灰度图，尺寸一致。
    """
    Tg, kg, Dg = 0.9994, -15, 0.5
    Ta, ka, Da = 0.9879, -22, 0.8

    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

    def _flip_conv(k, data):
        k = np.flip(k)
        data = np.pad(data, 1, mode='constant')
        return convolve2d(data, k, mode='valid')

    def _grad(img):
        img = img.astype(np.float64)
        sx = _flip_conv(h3, img)
        sy = _flip_conv(h1, img)
        g = np.sqrt(sx * sx + sy * sy)
        a = np.where(sx == 0, np.pi / 2, np.arctan(sy / sx))
        return g, a

    gA, aA = _grad(ir)
    gB, aB = _grad(vis)
    gF, aF = _grad(fused)

    def _q(aS, gS, aF_, gF_):
        GAF = np.where(gS > gF_, gF_ / (gS + 1e-12),
                       np.where(gS == gF_, gF_, gS / (gF_ + 1e-12)))
        AAF = 1 - np.abs(aS - aF_) / (math.pi / 2)
        Qg = Tg / (1 + np.exp(kg * (GAF - Dg)))
        Qa = Ta / (1 + np.exp(ka * (AAF - Da)))
        return Qg * Qa

    QAF = _q(aA, gA, aF, gF)
    QBF = _q(aB, gB, aF, gF)

    deno = np.sum(gA + gB) + 1e-12
    return float(np.sum(QAF * gA + QBF * gB) / deno)
