"""
工具函数模块

包含：
  - AverageMeter：指标均值计算器
  - save_model / load_model：模型保存与加载
  - setup_logger：日志配置
"""

import logging
import os
import torch
import torch.nn as nn


class AverageMeter:
    """运行均值计算器"""
    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f'{self.name}: {self.avg:.4f}'


def save_model(path: str, epoch: int, best_loss: float,
               model: nn.Module, optimizer=None):
    """保存模型 checkpoint"""
    state = {
        'epoch': epoch,
        'best_loss': best_loss,
        'model': model.state_dict(),
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, path)


def load_model(path: str, model: nn.Module,
               optimizer=None, device=None) -> tuple:
    """
    加载模型 checkpoint

    返回:
        (start_epoch, best_loss)
    """
    if device is None:
        device = torch.device('cpu')
    ckpt = torch.load(path, map_location=device)
    
    # 获取预训练权重的字典
    pretrained_dict = ckpt['model']
    model_dict = model.state_dict()
    
    # 筛选出键名相同且形状完全一致的权重
    valid_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    model_dict.update(valid_dict)
    model.load_state_dict(model_dict)
    
    if optimizer is not None and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception:
            pass # 忽略优化器异常（网络结构改动后优化器动量维数往往不匹配）
    epoch = ckpt.get('epoch', 0) + 1
    best_loss = ckpt.get('best_loss', float('inf'))
    return epoch, best_loss


def setup_logger(log_path: str, name: str = 'LMAFusion') -> logging.Logger:
    """配置日志记录器（同时输出到文件和控制台）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 文件 handler
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
