"""
Nabf — 融合伪影度量 (noise-based artifacts, 越低越好)
精简自 IVIF_ZOO-main/Metric/Nabf.py，去掉 torch 依赖。
"""
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


def _sobel(x):
    vt = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8
    ht = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / 8
    x_ext = _per_extn(x, 3)
    gv = convolve2d(x_ext, vt, mode='valid')
    gh = convolve2d(x_ext, ht, mode='valid')
    return gv, gh


def compute_nabf(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> float:
    """
    计算 Nabf（越低越好，≥0）。
    输入均为 2-D 灰度图，尺寸一致。
    """
    Td, Lg = 2, 1.5
    wt_min = 0.001
    Nrg, kg, sigmag = 0.9999, 19, 0.5
    Nra, ka, sigmaa = 0.9995, 22, 0.5

    x1 = ir.astype(np.float64)
    x2 = vis.astype(np.float64)
    xf = fused.astype(np.float64)
    p, q = xf.shape

    gvA, ghA = _sobel(x1)
    gA = np.sqrt(ghA ** 2 + gvA ** 2)
    gvB, ghB = _sobel(x2)
    gB = np.sqrt(ghB ** 2 + gvB ** 2)
    gvF, ghF = _sobel(xf)
    gF = np.sqrt(ghF ** 2 + gvF ** 2)

    def _ratio(gS, gF_):
        mask = (gS == 0) | (gF_ == 0)
        r = np.zeros_like(gS)
        with np.errstate(divide='ignore', invalid='ignore'):
            r[~mask] = np.where(gS > gF_, gF_ / gS, gS / gF_)[~mask]
        return r

    gAF = _ratio(gA, gF)
    gBF = _ratio(gB, gF)

    aA = np.where((gvA == 0) & (ghA == 0), 0, np.arctan(gvA / (ghA + 1e-30)))
    aB = np.where((gvB == 0) & (ghB == 0), 0, np.arctan(gvB / (ghB + 1e-30)))
    aF = np.where((gvF == 0) & (ghF == 0), 0, np.arctan(gvF / (ghF + 1e-30)))

    aAF = np.abs(np.abs(aA - aF) - np.pi / 2) * 2 / np.pi
    aBF = np.abs(np.abs(aB - aF) - np.pi / 2) * 2 / np.pi

    QAF = np.sqrt(Nrg / (1 + np.exp(-kg * (gAF - sigmag))) *
                  Nra / (1 + np.exp(-ka * (aAF - sigmaa))))
    QBF = np.sqrt(Nrg / (1 + np.exp(-kg * (gBF - sigmag))) *
                  Nra / (1 + np.exp(-ka * (aBF - sigmaa))))

    wtA = np.where(gA >= Td, gA ** Lg, 0.0)
    wtB = np.where(gB >= Td, gB ** Lg, 0.0)
    wt_sum = np.sum(wtA + wtB) + 1e-12

    na = np.where((gF > gA) & (gF > gB), 1.0, 0.0)
    NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum
    return float(NABF)
