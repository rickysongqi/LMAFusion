"""
精确裁剪计算 + 验证
根据两帧图像中无人机在 IR 和 VIS 中的位置，
反推出需要从 VIS 裁剪哪个区域才能与 IR 的FOV对应。
"""

import cv2
import numpy as np
from pathlib import Path
import os

DATA_DIR = './data_clean/val'
OUT_DIR  = './align_debug'
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 从诊断图观察到的无人机位置（在两张原图中的像素坐标）
# 图像: 20190925_101846_1_100075.png
# IR  640x369: 无人机中心约 (225, 255)
# VIS 1920x778: 无人机中心约 (890, 268)
# ────────────────────────────────────────────────────────────────
IR_W, IR_H = 640, 369
VIS_W, VIS_H = 1920, 778

drone_ir  = (225, 255)   # (x, y) in IR
drone_vis = (890, 268)   # (x, y) in VIS

# 期望无人机在裁剪后 VIS（resize 到 IR 尺寸后）的归一化位置
# = 和在 IR 中的归一化位置相同
norm_x = drone_ir[0] / IR_W   # 0.352
norm_y = drone_ir[1] / IR_H   # 0.691

print(f"目标归一化位置: ({norm_x:.3f}, {norm_y:.3f})")

# 裁剪区域宽高比 = IR 宽高比
aspect = IR_W / IR_H  # 1.735

# 最大可用 crop_h（保证 y1>=0）
max_crop_h_by_top = drone_vis[1] / norm_y
# 最大可用 crop_h（保证 y2<=VIS_H）
max_crop_h_by_bot = (VIS_H - drone_vis[1]) / (1 - norm_y)
crop_h = int(min(max_crop_h_by_top, max_crop_h_by_bot, 400))  # 不超过 400
crop_w = int(crop_h * aspect)

y1 = int(drone_vis[1] - norm_y * crop_h)
y2 = y1 + crop_h
x1 = int(drone_vis[0] - norm_x * crop_w)
x2 = x1 + crop_w

# 修正越界
x1 = max(0, x1); x2 = min(VIS_W, x2)
y1 = max(0, y1); y2 = min(VIS_H, y2)

print(f"计算得到 VIS_CROP = ({x1}, {y1}, {x2}, {y2})")
print(f"裁剪尺寸: {x2-x1} x {y2-y1}")
print(f"验证：无人机归一化位置应为 ({norm_x:.3f}, {norm_y:.3f})")
print(f"  实际：x={(drone_vis[0]-x1)/(x2-x1):.3f}, y={(drone_vis[1]-y1)/(y2-y1):.3f}")


def overlay_edges_zoom(ir, vis_crop):
    h, w = ir.shape
    vc = cv2.resize(vis_crop, (w, h))
    e_ir  = cv2.Sobel(ir, cv2.CV_32F, 1, 1, ksize=3)
    e_vis = cv2.Sobel(vc, cv2.CV_32F, 1, 1, ksize=3)
    e_ir  = cv2.convertScaleAbs(e_ir)
    e_vis = cv2.convertScaleAbs(e_vis)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:,:,1] = e_ir    # green = IR
    canvas[:,:,2] = e_vis   # red   = VIS
    base = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.4, canvas, 0.9, 0)


ir_files = sorted(Path(DATA_DIR + '/ir').glob('*.png'))[:6]
for ir_path in ir_files:
    vis_path = Path(DATA_DIR + '/vis') / ir_path.name
    if not vis_path.exists():
        continue
    ir  = cv2.imread(str(ir_path),  cv2.IMREAD_GRAYSCALE)
    vis = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)

    vis_cropped = vis[y1:y2, x1:x2]

    # 对比：左=无裁剪（旧方法），右=裁剪后（新方法）
    ov_old = overlay_edges_zoom(ir, vis)
    ov_new = overlay_edges_zoom(ir, vis_cropped)

    DISP_W, DISP_H = 700, 400
    ov_old = cv2.resize(ov_old, (DISP_W, DISP_H))
    ov_new = cv2.resize(ov_new, (DISP_W, DISP_H))

    cv2.putText(ov_old, 'BEFORE: Full VIS resize (ghosting)', (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    cv2.putText(ov_new, f'AFTER: crop({x1},{y1},{x2},{y2})', (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,100), 2)
    cv2.putText(ov_new, 'Green=IR, Red=VIS -- yellow=aligned', (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)

    row = np.concatenate([ov_old, ov_new], axis=1)
    out = os.path.join(OUT_DIR, 'crop_compare_' + ir_path.name)
    cv2.imwrite(out, row)
    print(f"Saved: {out}")
