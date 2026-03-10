# LMAFusion — 面向反无人机场景的轻量化红外与可见光图像融合

## 项目简介

本项目实现了一套高度创新的多模态图像融合网络 **LMAFusion**（Lightweight Mamba-guided Adaptive Fusion），
专为反无人机探测场景（低慢小目标、双光非严格对齐）设计。

## 核心模块

| 模块           | 文件                              | 功能                    |
| -------------- | --------------------------------- | ----------------------- |
| 🔧 DCN 形变对齐 | `modules/deformable_align.py`     | 补偿双光相机视差偏移    |
| 🔄 通道交换     | `modules/channel_exchange.py`     | 零 FLOPs 跨模态信息渗透 |
| 🔭 Mamba SSM 块 | `modules/mamba_block.py`          | O(N) 线性复杂度全局感知 |
| 🎛️ 双向门控融合 | `modules/bidirectional_gating.py` | 正反向抑制伪目标与泛光  |

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据集（将无人机数据集转换格式）
```bash
python prepare_data.py
# 可选: --max_sample 1000 只使用前 1000 对（快速测试）
```

### 3. 验证网络（前向传播 + 参数量统计）
```bash
python net.py
```

### 4. 开始训练
```bash
python train.py --epoch 50 --batch_size 8 --base_ch 16
```

### 5. 推理测试
```bash
python test.py --model_path ./model/best.pth --output_dir ./results
```

### 6. 评估指标（PSNR/SSIM/MI/EN/SF/AG）
```bash
python evaluate.py --fused_dir ./results
```

或者使用一键脚本：
```
run.bat
```

## 目录结构

```
LMAFusion/
├── modules/
│   ├── deformable_align.py    # 模块1: DCN 对齐
│   ├── channel_exchange.py    # 模块2: 通道交换
│   ├── mamba_block.py         # 模块3: Mamba SSM
│   └── bidirectional_gating.py # 模块4: 双向门控
├── net.py          # 主网络 LMAFusion
├── losses.py       # 复合损失函数
├── dataset.py      # 数据加载
├── train.py        # 训练脚本
├── test.py         # 推理脚本
├── evaluate.py     # 指标评估
├── prepare_data.py # 数据集准备
├── utils.py        # 工具函数
├── run.bat         # 一键启动（Windows）
└── requirements.txt
```

## 主要参数

| 参数           | 说明               | 默认值 |
| -------------- | ------------------ | ------ |
| `--base_ch`    | 基础通道数         | 16     |
| `--d_state`    | Mamba SSM 状态维度 | 16     |
| `--exchange_p` | 通道交换比例       | 0.5    |
| `--epoch`      | 训练轮数           | 50     |
| `--batch_size` | 批大小             | 8      |
| `--patch_size` | 训练 patch 大小    | 128    |
| `--lr`         | 学习率             | 1e-3   |

## 预期指标（目标）

| 指标   | 目标值  |
| ------ | ------- |
| 参数量 | < 200K  |
| GFLOPs | < 1.0   |
| PSNR   | > 40 dB |
| SSIM   | > 0.8   |
