# -*- coding: utf-8 -*-
"""
02_qc.py —— 读取抽取的分页，做质控，落成中间数据

产出
----
data/interim/stellar.pkl          全部分页合并后的分析用表（紧凑 dtype）
data/interim/qc_summary.csv       质控摘要
data/interim/plan_counts.csv      各计划/各观测类的计数与复观率
results/tables/qc_report.txt      人读报告
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import loader  # noqa: E402

ROOT = loader.ROOT
INTERIM = loader.INTERIM_DIR
TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(INTERIM, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

LOG = []


def say(s=''):
    print(s)
    LOG.append(str(s))


def main():
    t0 = time.time()
    say('=' * 76)
    say('  LAMOST 复观恒星 · 数据质控')
    say('=' * 76)

    files = loader.page_files()
    say('分页文件: %d 个' % len(files))
    if not files:
        say('!! 没有分页数据，请先运行 run_extract.py')
        return 1
    say('预计行数: %d' % (len(files) * 20000))

    df = loader.load_pages()
    df = loader.to_numeric(df)
    df = loader.add_colors(df)
    say('载入后: %d 行 × %d 列' % df.shape)
    say('耗时 %.1f 分' % ((time.time() - t0) / 60.0))

    # ---------------- 基础计数 ----------------
    say('\n' + '-' * 76)
    say('一、规模与复观倍率')
    say('-' * 76)
    n_obs = len(df)
    n_obsid = int(df['obsid'].nunique())
    n_gp = int(df['gp_id'].nunique())
    n_des = int(df['designation'].nunique())
    say('  观测总数            %10d' % n_obs)
    say('  唯一 obsid          %10d   (应等于观测总数: %s)'
        % (n_obsid, '是' if n_obsid == n_obs else '否 !!'))
    say('  唯一 gp_id          %10d' % n_gp)
    say('  唯一 designation    %10d' % n_des)
    say('  复观次数(gp_id)     %10d   复观率 %.3f×' % (n_obs - n_gp, n_obs / max(n_gp, 1)))
    say('  复观次数(desig)     %10d   复观率 %.3f×' % (n_obs - n_des, n_obs / max(n_des, 1)))

    # ---------------- 哨兵/缺失 ----------------
    say('\n' + '-' * 76)
    say('二、哨兵与缺失（LAMOST 用大负数表示无效，本项目将 <= -900 置 NaN）')
    say('-' * 76)
    rows = []
    for c in loader.PHYS_COLS + ['g_r', 'r_i', 'i_z', 'z_y']:
        if c in df.columns:
            m = int(df[c].isna().sum())
            rows.append({'column': c, 'missing': m, 'pct': 100.0 * m / n_obs})
    qc = pd.DataFrame(rows).sort_values('pct', ascending=False)
    say('  %-18s %12s %9s' % ('列', '缺失', '占比'))
    for _, r in qc.head(14).iterrows():
        say('  %-18s %12d %8.2f%%' % (r['column'], r['missing'], r['pct']))
    qc.to_csv(os.path.join(INTERIM, 'qc_summary.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 计划分类 ----------------
    say('\n' + '-' * 76)
    say('三、观测计划分类（planid 首段字母）')
    say('-' * 76)
    df['plan_prefix'] = df['planid'].astype('string').str.extract(r'^([A-Za-z]+)',
                                                                  expand=False)
    df['plan_prefix'] = df['plan_prefix'].fillna('UNK')
    g = df.groupby('plan_prefix', observed=True).agg(
        n_obs=('obsid', 'size'),
        n_star=('gp_id', 'nunique'),
        n_desig=('designation', 'nunique'),
    )
    g['repeat_rate'] = g['n_obs'] / g['n_star']
    g['repeat_obs'] = g['n_obs'] - g['n_star']
    g = g.sort_values('n_obs', ascending=False)
    say('  %-6s %12s %12s %10s %10s %12s' %
        ('前缀', '观测数', '唯一星', '复观率', '复观次数', '示例planid'))
    ex = df.groupby('plan_prefix', observed=True)['planid'].first()
    for p, r in g.iterrows():
        say('  %-6s %12d %12d %9.3fx %10d   %s'
            % (p, r['n_obs'], r['n_star'], r['repeat_rate'], r['repeat_obs'], ex.get(p, '')))
    g.to_csv(os.path.join(INTERIM, 'plan_counts.csv'), encoding='utf-8-sig')

    # ---------------- 复观组规模分布 ----------------
    say('\n' + '-' * 76)
    say('四、复观组规模分布（按 gp_id）')
    say('-' * 76)
    size = df.groupby('gp_id', observed=True).size()
    dist = size.value_counts().sort_index()
    tot_star = len(size)
    say('  %8s %14s %12s %14s' % ('每组观测数', '恒星数', '占比', '占全部观测'))
    cum = 0
    for k, v in dist.items():
        cum += int(k) * int(v)
        if k <= 12 or v > tot_star * 0.001:
            say('  %8d %14d %11.3f%% %13.2f%%'
                % (k, v, 100.0 * v / tot_star, 100.0 * cum / n_obs))
    say('  合计恒星 %d，其中复观星(>=2 次) %d (%.1f%%)'
        % (tot_star, int((size >= 2).sum()), 100.0 * (size >= 2).sum() / tot_star))

    # ---------------- 保留复观子集 ----------------
    say('\n' + '-' * 76)
    say('五、保存中间数据')
    say('-' * 76)
    for c in [c for c in df.columns if df[c].dtype.kind == 'f']:
        df[c] = df[c].astype('float32')
    for c in ('planid', 'subclass', 'designation', 'plan_prefix'):
        if c in df.columns:
            df[c] = df[c].astype('category')
    out = os.path.join(INTERIM, 'stellar.pkl')
    df.to_pickle(out, protocol=4)
    say('  已写 %s (%.1f MB)' % (out, os.path.getsize(out) / 1048576.0))

    size = df.groupby('gp_id', observed=True).size()
    rep = df[df['gp_id'].isin(size[size >= 2].index)]
    outr = os.path.join(INTERIM, 'stellar_repeat.pkl')
    rep.to_pickle(outr, protocol=4)
    say('  已写 %s (%d 行, %.1f MB)'
        % (outr, len(rep), os.path.getsize(outr) / 1048576.0))

    say('\n  复观子集: %d 行 / %d 颗星 = %.1f%% 的观测'
        % (len(rep), size[size >= 2].size, 100.0 * len(rep) / n_obs))
    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))

    with open(os.path.join(TABLES, 'qc_report.txt'), 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
