# -*- coding: utf-8 -*-
"""
统一路径解析 —— 所有脚本与模块都应从这里取路径，不要硬编码绝对路径。

为什么要这样
------------
早期版本的 10 个脚本把项目根写成字面量
`ROOT = r'D:\\ds工作区\\01-科研实习\\LAMOST-复观恒星'`，
后果有两个：

1. **不可移植**：别人 clone 到别的盘符/目录就跑不起来。
2. **无法安全测试**：任何"复制项目到临时目录再试跑"的做法都会因为脚本仍然
   读写原始路径而失效（本项目的一轮独立验证就撞上了这个坑）。

本模块把根目录解析为**本文件所在位置向上两级**（`src/` 的父目录），
因此整个项目可以整体搬移。如需覆盖，设置环境变量 `LAMOST_ROOT`。
"""
from __future__ import annotations

import os

# src/paths.py → src/ → 项目根
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('LAMOST_ROOT', _DEFAULT_ROOT)

DATA = os.path.join(ROOT, 'data')
RAW = os.path.join(DATA, 'raw')
INTERIM = os.path.join(DATA, 'interim')
RESULTS = os.path.join(ROOT, 'results')
TABLES = os.path.join(RESULTS, 'tables')
FIGURES = os.path.join(RESULTS, 'figures')
REPORT = os.path.join(RESULTS, 'REPORT.md')

# 抽取分页与光谱缓存的位置（供脚本引用，避免各自拼路径）
PAGES_DIR = os.path.join(RAW, 'stellar_pages')
PROGRESS = os.path.join(RAW, 'progress.json')


def makedirs(*paths):
    """确保这些目录存在。"""
    for p in paths:
        os.makedirs(p, exist_ok=True)
