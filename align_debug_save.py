"""
静默版对齐诊断 - 不弹窗，直接把诊断图存到 align_debug/ 目录
"""
import cv2
import numpy as np
from pathlib import Path
import os

DATA_DIR = './data_clean/val'
OUT_DIR  = './align_debug'
NUM_SHOW = 4

os.makedirs(OUT_DIR, exist_ok=True)

def sobel(img):
    g = cv2.Sobel(img, cv2.CV_32F, 1, 1, ksize=3)
    return cv2.convertScaleAbs(g)

def overlay_edges(ir, vis_for_overlay):
    h, w = ir.shape[:2]
    vis_rs = cv2.resize(vis_for_overlay, (w, h))
    edge_ir  = sobel(ir)
    edge_vis = sobel(vis_rs)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :, 1] = edge_ir
    canvas[:, :, 2] = edge_vis
    base = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.4, canvas, 0.8, 0)

ir_files = sorted(Path(DATA_DIR + '/ir').glob('*.png'))[:NUM_SHOW]

for ir_path in ir_files:
    vis_path = Path(DATA_DIR + '/vis') / ir_path.name
    if not vis_path.exists():
        continue

    ir  = cv2.imread(str(ir_path),  cv2.IMREAD_GRAYSCALE)
    vis = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
    ir_h, ir_w   = ir.shape
    vis_h, vis_w = vis.shape

    # 缩放到同一显示高度用于并排展示
    DISP_H = 400
    ir_disp  = cv2.resize(ir,  (int(ir_w  * DISP_H / ir_h),  DISP_H))
    vis_disp = cv2.resize(vis, (int(vis_w * DISP_H / vis_h), DISP_H))

    # 上半部分：原图并排（带标注）
    ir_bgr  = cv2.cvtColor(ir_disp,  cv2.COLOR_GRAY2BGR)
    vis_bgr = cv2.cvtColor(vis_disp, cv2.COLOR_GRAY2BGR)
    cv2.putText(ir_bgr,  f'IR  {ir_w}x{ir_h}',  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(vis_bgr, f'VIS {vis_w}x{vis_h}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)
    raw_row = np.concatenate([ir_bgr, vis_bgr], axis=1)

    # 下半部分：4种裁剪方案叠加边缘图
    crops = [
        (None,                    "no_crop (full VIS resize)"),
        ((200,0,1336,576),        "x1=200 x2=1336 (slight trim)"),
        ((350,0,1186,576),        "x1=350 x2=1186 (medium)"),
        ((450,0,1100,576),        "x1=450 x2=1100 (aggressive)"),
    ]

    panels = []
    for crop, label in crops:
        if crop is not None:
            x1,y1,x2,y2 = crop
            vis_c = vis[y1:y2, x1:x2]
        else:
            vis_c = vis
        ov = overlay_edges(ir, vis_c)
        ov = cv2.resize(ov, (DISP_H * 2, DISP_H // 2))  # 固定宽高
        cv2.putText(ov, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        panels.append(ov)

    # 两行两列排版
    crop_grid = np.concatenate([
        np.concatenate(panels[:2], axis=1),
        np.concatenate(panels[2:], axis=1),
    ], axis=0)

    # 合并两行
    W = max(raw_row.shape[1], crop_grid.shape[1])
    def pad_w(img, target_w):
        if img.shape[1] < target_w:
            pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
            return np.concatenate([img, pad], axis=1)
        return img

    out = np.concatenate([pad_w(raw_row, W), pad_w(crop_grid, W)], axis=0)
    out_path = os.path.join(OUT_DIR, ir_path.name)
    cv2.imwrite(out_path, out)
    print(f"Saved: {out_path}")

print(f"\n全部保存到 {OUT_DIR}/")
print("绿色轮廓 = IR边缘，红色轮廓 = VIS边缘")
print("两色叠合变黄色 = 对齐良好")
