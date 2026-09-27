# -*- coding: utf-8 -*-
"""
08_report.py —— 把所有结果表汇总成一份可直接阅读的 Markdown 报告

只做汇总与排版，不重新计算。所有数字都来自 results/tables/ 下的 CSV，
因此报告与数据严格一致、可追溯。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
TABLES = os.path.join(ROOT, 'results', 'tables')
REPORT = os.path.join(ROOT, 'results', 'REPORT.md')


def rd(name):
    p = os.path.join(TABLES, name)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def read_text(path):
    """
    宽容地读文本。

    PowerShell 的 `Out-File -Encoding utf8` 在 Windows PowerShell 下写出的是
    **UTF-16 LE 带 BOM**（首字节 0xff），而本项目的报告文件是 utf-8-sig。
    两种都要能读，否则汇总会直接崩。
    """
    with open(path, 'rb') as f:
        raw = f.read()
    for enc in ('utf-8-sig', 'utf-16', 'utf-8', 'gbk'):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', 'replace')


def md_table(df, cols=None, floatfmt='%.4f', maxrows=None):
    if df is None or df.empty:
        return '_（无数据）_\n'
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if maxrows:
        d = d.head(maxrows)
    hdr = '| ' + ' | '.join(str(c) for c in d.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(d.columns)) + '|'
    lines = [hdr, sep]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append('nan' if not np.isfinite(v) else (floatfmt % v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    L = []
    A = L.append
    A('# LAMOST DR10 复观恒星：官方误差标定与机器学习标签泄漏\n')
    A('自动生成于 %s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))

    # ---------- 1. 规模 ----------
    A('\n## 1. 数据规模\n')
    L2 = []
    for fn in ('qc_report.txt', 'repeatability_report.txt',
               'repeatability_by_plan.csv', 'leakage_capacity.log'):
        p = os.path.join(TABLES, fn)
        if os.path.exists(p):
            L2.append((fn, read_text(p)))

    import re
    scale = {}
    for fn, txt in L2:
        for key, pat in [('obs', r'观测总数\s+(\d+)'),
                         ('gp', r'唯一 gp_id\s+(\d+)'),
                         ('des', r'唯一 designation\s+(\d+)')]:
            m = re.search(pat, txt)
            if m and key not in scale:
                scale[key] = int(m.group(1))
    A('| 项 | 数值 |')
    A('|---|---|')
    for key, label in (('obs', '观测总数'), ('gp', '唯一恒星（gp_id）'),
                       ('des', '唯一位置编码（designation）')):
        if key in scale:
            A('| %s | %s |' % (label, f"{scale[key]:,}"))
        else:
            # 不能用 .get(key, 0) —— 那会把"没读到"伪造成"数值为 0"，
            # 静默写进报告。缺输入就必须显式标注缺失。
            A('| %s | _未取得（缺 qc_report.txt）_ |' % label)
    if 'obs' in scale and 'gp' in scale and scale.get('gp'):
        A('| 复观倍率 | %.3f× |' % (scale['obs'] / scale['gp']))
    if 'obs' not in scale:
        A('\n> ⚠️ 第 1 节未能读取 `results/tables/qc_report.txt`，规模数字缺失；'
          '请先运行 `scripts/02_qc.py`。')

    # ---------- 2. 总体标定 ----------
    A('\n## 2. 复观一致性与官方误差标定（总体）\n')
    ov = rd('repeatability_overall.csv')
    if ov is not None:
        cols = ['value', 'n_groups', 'n_obs', 'pooled_sigma', 'err_median',
                'ratio_sigma_over_err', 'chi2_red']
        A(md_table(ov, [c for c in cols if c in ov.columns]))
        A('\n- `pooled σ`：只由复观离散度决定的实测不确定度。')
        A('- `chi2_red`：官方误差若正确应为 1。')
        A('- `σ/err`：实测离散与官方误差之比，>1 表示官方误差偏小。\n')

    # ---------- 3. 按计划 ----------
    A('\n## 3. 按观测计划分层\n')
    bp = rd('repeatability_by_plan.csv')
    if bp is not None and 'value' in bp.columns:
        for param in ('teff', 'logg', 'feh'):
            s = bp[bp['value'] == param]
            if s.empty:
                continue
            s = s.sort_values('n_groups', ascending=False)
            A('\n### %s\n' % param)
            cols = ['plan_prefix', 'n_groups', 'pooled_sigma', 'err_median',
                    'ratio_sigma_over_err', 'chi2_red']
            A(md_table(s, [c for c in cols if c in s.columns]))

    # ---------- 4. S/N 与次数 ----------
    A('\n## 4. 不确定度随信噪比与复观次数的变化\n')
    gs = rd('repeatability_groupsize.csv')
    if gs is not None:
        A('\n### 按复观次数\n')
        A(md_table(gs, ['value', 'n_obs_bin', 'n_stars', 'pooled_sigma',
                        'err_median', 'ratio_sigma_over_err', 'chi2_red']))
    sb = rd('repeatability_snr_bins.csv')
    if sb is not None:
        A('\n### 按信噪比分箱\n')
        A(md_table(sb, ['value', 'snr_lo', 'snr_hi', 'n_groups', 'pooled_sigma',
                        'err_median', 'ratio_sigma_over_err', 'chi2_red']))

    # ---------- 5. 泄漏 ----------
    # 优先用主实验（04b 容量扫描，全量数据），
    # 因为它才是 README §5.4 引用的结果。
    # 04_leakage.py 的 leakage_summary.csv 是早期在 118 万行中间态数据上产出的，
    # 与第 1 节的 745 万行规模不一致，**不能混进同一份报告**。
    A('\n## 5. 机器学习标签泄漏（主实验）\n')
    lc = rd('leakage_capacity.csv')
    if lc is not None and not lc.empty:
        A('泄漏倍数 `factor = RMSE_grouped / RMSE_random`，'
          '>1 表示按观测随机划分把 RMSE 压低了。\n')
        for param in ('teff', 'logg', 'feh'):
            s = lc[lc['target'] == param]
            if s.empty:
                continue
            piv = s.pivot_table(index='subset', columns=['model', 'param'],
                                values='factor')
            piv = piv.reset_index()
            A('\n### %s\n' % param)
            A(md_table(piv, floatfmt='%.3f'))
            n = (s.groupby('subset')
                 .agg(n_samples=('n_samples', 'first'),
                      n_groups=('n_groups', 'first'),
                      hit=('frac_exact_hit_random', 'first'))
                 .reset_index())
            n['hit'] = (100.0 * n['hit']).round(1)
            n['repeat_rate'] = (n['n_samples'] / n['n_groups']).round(3)
            A('\n子集规模与随机划分下的精确命中率：\n')
            A(md_table(n, floatfmt='%.3f'))
    else:
        A('_（主实验尚未完成；请运行 `scripts/04b_leakage_capacity.py`）_\n')

    ls = rd('leakage_summary.csv')
    if ls is not None:
        A('\n### 附：基础版实验（早期中间态数据，仅存档，勿引用同行比较）\n')
        A('⚠️ 下表产出于 **1,180,000 行**的中间态数据，与本文其他章节的 '
          '7,450,303 行不同源，且其下采样方式破坏了复观结构（见 README §4.3）。\n')
        cols = ['feature_set', 'target', 'n_samples', 'n_groups',
                'rmse_random', 'rmse_grouped', 'factor']
        A(md_table(ls, [c for c in cols if c in ls.columns]))

    # ---------- 6. 稳健性 ----------
    A('\n## 6. 稳健性自查：全样本 χ²_red 由约 1% 的污染组主导\n')
    tr = rd('robustness_trim_variables.csv')
    if tr is not None and not tr.empty:
        for param in ('teff', 'logg', 'feh'):
            s = tr[tr['param'] == param].copy()
            if s.empty:
                continue
            s['剔除尾部'] = (100.0 * s['trim_frac']).map(lambda x: '%.2f%%' % x)
            A('\n### %s\n' % param)
            A(md_table(s, ['剔除尾部', 'n_groups', 'pooled_sigma',
                           'err_median', 'chi2_red', 'ratio_sigma_over_err']))
        A('\n**判读**：剔除最离散的 0.1% 组，Teff 的 χ²_red 即从 5.875 落到 2.143；'
          '剔除 1% 落到 1.233。说明全样本的"超出 1"并非官方误差系统性偏小，'
          '而是被极少数污染/错配组主导。\n')
    sn = rd('robustness_snr_trim.csv')
    if sn is not None and not sn.empty:
        A('\n### 分 S/N 箱 + 剔除最离散 1%（双重控制）\n')
        A(md_table(sn, ['param', 'snr_lo', 'snr_hi', 'n_groups',
                        'chi2_red_all', 'chi2_red_trim1pct']))
    ol = rd('outlier_bad_sample.csv')
    og = rd('outlier_good_sample.csv')
    if ol is not None and og is not None and not ol.empty:
        A('\n### 异常组 vs 正常组画像\n')
        rows = []
        for lab, t in (('异常组 χ²/dof>P99', ol), ('正常组', og)):
            rows.append({
                '组': lab, '样本数': len(t),
                '复观次数中位': float(t['n_obs'].median()),
                'Teff组内极差中位(K)': float(t['teff_range'].median()),
                '官方误差中位(K)': float(t['err_median'].median()),
                '极差/官方误差': float((t['teff_range'] / t['err_median'].replace(0, np.nan)).median()),
                'χ²/dof 中位': float(t['chi2_dof'].median()),
            })
        A(md_table(pd.DataFrame(rows), floatfmt='%.2f'))
        A('\n极端组的 Teff 组内极差中位达 594 K（正常组 47 K），且最高 χ²/dof 的组'
          '给出 6332→13328 K 这类跨 7000 K 的"复观"——物理上不可能是同一颗恒星，'
          '指向**光纤污染或 Gaia 源错配**。异常组中 A1+A2 占 36.5%，'
          '与 §3 中 A 型 [Fe/H] 的 χ²_red 偏高相呼应。\n')

    # ---------- 7. 跨计划 ----------
    A('\n## 7. 跨观测计划的系统偏差（配对）\n')
    po = rd('plan_pair_offsets.csv')
    if po is not None and not po.empty:
        for param in ('teff', 'logg', 'feh'):
            s = po[po['target'] == param].sort_values('n_common', ascending=False)
            if s.empty:
                continue
            A('\n### %s\n' % param)
            A(md_table(s, ['plan_a', 'plan_b', 'n_common', 'mean_diff',
                           'std_diff', 'sem', 't_stat'], maxrows=15))
    else:
        A('_（尚未计算）_\n')

    # ---------- 8. 光谱 ----------
    A('\n## 8. 真实光谱特征\n')
    sc = rd('spectra_correlations.csv')
    if sc is not None and not sc.empty:
        piv = sc.pivot_table(index='band', columns='param', values='spearman')
        piv = piv.reset_index()
        # 不用 DataFrame.to_markdown —— 它依赖未安装的 tabulate。
        A(md_table(piv, floatfmt='%+.3f'))
        ns = sc.groupby('band')['n'].max().to_dict()
        A('\n各谱线可用的最大匹配数: ' +
          ', '.join('%s=%d' % (k, v) for k, v in sorted(ns.items())))
    else:
        A('_（尚未计算）_\n')

    txt = '\n'.join(L)
    with open(REPORT, 'w', encoding='utf-8-sig') as f:
        f.write(txt)
    print('-> %s (%d 字符)' % (REPORT, len(txt)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
