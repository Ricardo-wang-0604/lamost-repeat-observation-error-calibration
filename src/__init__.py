# -*- coding: utf-8 -*-
"""LAMOST 复观恒星：重复观测不确定度与机器学习标签泄漏

子模块
------
tap        LAMOST TAP(IVOA) 客户端：VOTable 解析、游标分页
loader     分页读取、数值化、哨兵处理、颜色指数、质控摘要
repeat     复观一致性与官方误差标定（pooled σ / 归一化卡方 / S/N 分箱）
leakage    观测级随机划分 vs 恒星级分组划分的标签泄漏实验
spectra    LAMOST FITS 光谱读取与谱线指数
plots      绘图
"""
__version__ = '0.1.0'
