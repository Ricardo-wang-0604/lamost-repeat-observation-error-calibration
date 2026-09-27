# -*- coding: utf-8 -*-
"""
数据加载与质控

关键约定
--------
分组键（三选一，本项目以 gp_id 为主、designation 为对照）：
  * obsid       : 单次观测的唯一编号（已验证 count(distinct)==count(*)）
  * designation : 位置编码 JHHMMSS.ss+DDMMSS.s，同一颗星多次观测共享
  * gp_id       : Gaia 源编号，物理恒星的唯一标识（数值型，便于聚合）

缺失值哨兵：LAMOST 用大负数表示无效，常见 -999 / -9999。
本项目统一把 <= -900 的物理量视为 NaN。
注意 alpha_m / vsini_lasp 在 DR10 stellar 中大量为 -9999，属"未测量"而非"零"。
"""
from __future__ import annotations

import glob
import gzip
import os

import numpy as np
import pandas as pd

from . import paths as _paths

ROOT = _paths.ROOT
RAW_DIR = _paths.RAW
PAGES_DIR = _paths.PAGES_DIR
INTERIM_DIR = _paths.INTERIM

SENTINEL_CUTOFF = -900.0

# 物理量列（其余为标识/分类）
PHYS_COLS = ['ra', 'dec', 'snrg', 'snrr',
             'mag_ps_g', 'mag_ps_r', 'mag_ps_i', 'mag_ps_z', 'mag_ps_y',
             'gaia_g_mean_mag',
             'teff', 'teff_err', 'logg', 'logg_err', 'feh', 'feh_err',
             'alpha_m', 'rv']

# 标识/分类列（不做哨兵替换）
ID_COLS = ['obsid', 'gp_id', 'designation', 'planid', 'spid', 'fiberid', 'mjd',
           'subclass', 'class', 'fibertype']

TARGETS = ['teff', 'logg', 'feh']

DTYPES = {
    'obsid': 'int64', 'gp_id': 'int64', 'designation': 'string',
    'mjd': 'int32', 'subclass': 'string',
}


def page_files():
    """
    收集所有分页文件。

    分片并行抽取后，数据在 data/raw/seg{0..N}/page_*.csv.gz。
    注意：worker 0 会把自己的旧成果从 stellar_pages 复制进 seg0，
    所以一旦 seg* 存在就**只读 seg***，否则会把同一批数据读两遍。
    """
    segs = sorted(glob.glob(os.path.join(RAW_DIR, 'seg*', 'page_*.csv.gz')))
    if segs:
        return segs
    return sorted(glob.glob(os.path.join(PAGES_DIR, 'page_*.csv.gz')))


def load_pages(limit=None, columns=None, verbose=True):
    """把抽取的所有分页读成单个 DataFrame。"""
    files = page_files()
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError('没有找到分页文件：%s' % PAGES_DIR)

    frames = []
    for i, p in enumerate(files):
        with gzip.open(p, 'rt', encoding='utf-8-sig') as fh:
            # 全部按字符串读入，再做数值转换，避免混合类型告警
            df = pd.read_csv(fh, dtype=str, keep_default_na=False)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        frames.append(df)
        if verbose and (i + 1) % 20 == 0:
            print('  已读 %d/%d 页' % (i + 1, len(files)))

    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print('  合并完成: %d 行 × %d 列' % out.shape)
    return out


def to_numeric(df, cols=None):
    """字符串 → 数值，并把哨兵值置为 NaN。"""
    cols = cols or [c for c in PHYS_COLS if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] <= SENTINEL_CUTOFF, c] = np.nan
    for c in ('obsid', 'gp_id', 'mjd'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    return df


def add_colors(df):
    """由 PS1 星等构造颜色指数与 Gaia-PS1 色差。"""
    pairs = [('g_r', 'mag_ps_g', 'mag_ps_r'), ('r_i', 'mag_ps_r', 'mag_ps_i'),
             ('i_z', 'mag_ps_i', 'mag_ps_z'), ('z_y', 'mag_ps_z', 'mag_ps_y'),
             ('g_i', 'mag_ps_g', 'mag_ps_i'), ('r_z', 'mag_ps_r', 'mag_ps_z')]
    for name, a, b in pairs:
        if a in df.columns and b in df.columns:
            df[name] = df[a] - df[b]
    if 'g_r' in df.columns and 'gaia_g_mean_mag' in df.columns:
        df['g_gaia'] = df['mag_ps_g'] - df['gaia_g_mean_mag']
    return df


def qc_summary(df, label=''):
    """打印质控摘要。"""
    n = len(df)
    print('\n[质控] %s  共 %d 行' % (label, n))
    rows = []
    for c in PHYS_COLS + ['g_r', 'r_i', 'i_z', 'z_y']:
        if c not in df.columns:
            continue
        miss = int(df[c].isna().sum())
        rows.append((c, miss, 100.0 * miss / n))
    rows.sort(key=lambda r: -r[2])
    print('  %-16s %12s %8s' % ('列', '缺失/哨兵', '占比'))
    for c, m, pct in rows[:12]:
        print('  %-16s %12d %7.2f%%' % (c, m, pct))
    if 'obsid' in df.columns:
        print('  obsid 唯一: %s | 唯一 gp_id: %d | 唯一 designation: %d' % (
            df['obsid'].is_unique, df['gp_id'].nunique(), df['designation'].nunique()))
    return rows


def repeat_groups(df, key='gp_id', min_obs=2):
    """返回复观组：同一 key 出现 >= min_obs 次。"""
    cnt = df.groupby(key, observed=True).size()
    reps = cnt[cnt >= min_obs].index
    mask = df[key].isin(reps)
    return mask, cnt


def sample_definitions(df, key='gp_id', min_obs=2):
    """构造『全样本 / 复观子样本 / 单次观测样本』。"""
    mask, cnt = repeat_groups(df, key=key, min_obs=min_obs)
    return {
        'all': df,
        'repeat': df[mask],
        'single': df[~mask],
        'group_size': cnt,
    }


if __name__ == '__main__':
    print('分页文件数:', len(page_files()))
