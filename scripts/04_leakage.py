# -*- coding: utf-8 -*-
"""
04_leakage.py —— 标签泄漏实验

设计
----
同一颗恒星（同一 gp_id）被 LAMOST 观测多次，每次一行样本。
其中**测光特征（PS1 griz y、Gaia G、颜色）在多次观测之间完全相同** —— 它们是恒星级常数，
而恒星参数标签（teff/logg/feh）因每次曝光信噪比不同而有涨落。

于是划分方式决定结果：

  按『观测』随机划分 → 同一颗星的特征向量同时出现在训练集与验证集
                    → 模型能"记住"这颗星 → 验证精度虚高
  按『恒星』分组划分 → 每颗星只出现在一侧 → 反映对未见过恒星的泛化能力

为区分"泄漏来自哪里"，本脚本对三套特征分别实验：

  A 测光（恒星级常数）   → 预期泄漏极其显著
  B S/N（观测级变量）     → 预期两组划分几乎无差别（本就不构成恒星级泄漏）
  C 测光 + S/N            → 介于两者之间

这与风电功率预测中"多场站 NWP 输入重复导致交叉验证泄漏"是同一类问题，
分组键从『场站』换成『恒星』。

产出
----
results/tables/leakage_runs.csv        每次重复的明细
results/tables/leakage_summary.csv     按 特征集 × 标签 汇总
results/tables/leakage_importance_*.csv 特征重要性
results/tables/leakage_report.txt      人读报告
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# LightGBM 用 numpy 训练时会自造 Column_0/Column_1 特征名，
# sklearn 的 check 随后会刷 UserWarning，属已知无害行为，压掉以免淹没输出。
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
sys.path.insert(0, ROOT)
from src import loader, leakage as lk  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LOG = []


def say(s=''):
    print(s)
    LOG.append(str(s))


def main():
    t0 = time.time()
    say('=' * 84)
    say('  LAMOST 复观恒星 · 机器学习标签泄漏实验')
    say('=' * 84)

    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    say('载入 %d 行' % len(df))

    df = df[df['gp_id'].notna()].copy()
    size = df.groupby('gp_id', observed=True).size()
    say('恒星 %d 颗，复观星 %d 颗 (%.2f%%)'
        % (len(size), int((size >= 2).sum()), 100.0 * (size >= 2).sum() / len(size)))

    FEATURE_SETS = ['photometry', 'obs', 'all']
    TARGETS = ['teff', 'logg', 'feh']
    REPEATS = 5

    runs, summaries = [], []

    for feat in FEATURE_SETS:
        for tgt in TARGETS:
            # 构造矩阵（内部已 dropna）
            try:
                X, y, g, cols, sub = lk.build_matrix(df, feature_set=feat, target=tgt)
            except Exception as exc:  # noqa: BLE001
                say('  [跳过] %s × %s : %s' % (feat, tgt, exc))
                continue
            if len(y) < 2000 or len(np.unique(g)) < 100:
                say('  [跳过] %s × %s : 有效样本不足 (%d 行 / %d 星)'
                    % (feat, tgt, len(y), len(np.unique(g))))
                continue

            say('\n' + '-' * 84)
            say('%s  ×  %s' % (feat, tgt))
            say('  特征 %d 个: %s' % (len(cols), ', '.join(cols)))
            say('  有效样本 %d 行 / %d 颗星  (重复率 %.3f×)'
                % (len(y), len(np.unique(g)), len(y) / len(np.unique(g))))

            # 复用已构造的矩阵（matrix=...），避免在 745 万行上重复执行 dropna。
            # 早期版本这一步会在泄漏实验内部再算一遍、特征重要性里算第三遍。
            M = (X, y, g, cols, sub)
            tab, summ, infl, _ = lk.leakage_experiment(
                df, target=tgt, feature_set=feat, kind='lgbm',
                test_size=0.2, repeats=REPEATS, seed=42, matrix=M)
            tab['feature_set'] = feat
            tab['target'] = tgt
            runs.append(tab)

            infl['feature_set'] = feat
            summaries.append(infl)

            say('  %-10s %10s %10s %10s' % ('划分', 'RMSE均值', 'RMSE标准差', 'R²均值'))
            for _, r in summ.iterrows():
                rr = tab[tab['tag'] == r['tag']]['r2'].mean()
                say('  %-10s %10.4f %10.4f %10.4f' % (r['tag'], r['mean'], r['std'], rr))
            say('  → RMSE 虚高倍数 %.3f×   相对虚高 %.1f%%'
                % (infl['factor'], infl['rel_inflation_pct']))

            imp = lk.feature_importance(df, target=tgt, feature_set=feat,
                                        kind='lgbm', matrix=M)
            if len(imp):
                imp.to_csv(os.path.join(TABLES, 'leakage_importance_%s_%s.csv' % (feat, tgt)),
                           index=False, encoding='utf-8-sig')

    if not runs:
        say('\n!! 没有任何实验跑成功')
        return 1

    RUNS = pd.concat(runs, ignore_index=True)
    SUMM = pd.DataFrame(summaries)
    RUNS.to_csv(os.path.join(TABLES, 'leakage_runs.csv'), index=False, encoding='utf-8-sig')
    SUMM.to_csv(os.path.join(TABLES, 'leakage_summary.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 汇总 ----------------
    say('\n' + '=' * 84)
    say('  汇总：随机划分 vs 分组划分')
    say('=' * 84)
    say('  %-11s %-6s %6s %10s %9s %10s %9s %8s %7s'
        % ('特征集', '标签', '特征数', '样本', '恒星数', 'RMSE随机', 'RMSE分组', '虚高倍数', 'R²随机'))
    for _, r in SUMM.iterrows():
        say('  %-11s %-6s %6d %10d %9d %10.4f %10.4f %8.3fx %7.3f'
            % (r['feature_set'], r['target'], r['features'], r['n_samples'],
               r['n_groups'], r['rmse_random'], r['rmse_grouped'], r['factor'],
               r['r2_random']))

    # ---------------- 交叉验证：随机划分 ≈ 复观池化 σ ----------------
    say('\n' + '=' * 84)
    say('  交叉验证：随机划分的 RMSE 是否等于复观池化 σ')
    say('=' * 84)
    say('  机理：随机划分下，测试集里的每颗星在训练集里都出现过。')
    say('        模型对该星只能给出一个预测值，最优策略就是给出这颗星的均值。')
    say('        于是 RMSE_random → 组内离散度 = pooled σ。')
    say('        若两者相等，就证明随机划分测的是"重复预测已见过的星"，')
    say('        而不是"预测未见过的新星"的泛化能力。')
    ovp = os.path.join(TABLES, 'repeatability_overall.csv')
    if os.path.exists(ovp):
        ov = pd.read_csv(ovp)
        say('\n  %-8s %14s %16s %10s'
            % ('标签', 'RMSE_random', 'pooled σ', '比值'))
        for _, r in SUMM[SUMM['feature_set'] == 'photometry'].iterrows():
            m = ov[ov['value'] == r['target']]
            if m.empty:
                continue
            sg = float(m['pooled_sigma'].iloc[0])
            say('  %-8s %14.4f %16.4f %10.3f'
                % (r['target'], r['rmse_random'], sg,
                   r['rmse_random'] / sg if sg else np.nan))
        say('\n  比值接近 1 即证实上述机理。')
    else:
        say('  （缺 repeatability_overall.csv，跳过）')

    say('\n  解读：')
    for _, r in SUMM.iterrows():
        if r['feature_set'] == 'photometry':
            say('    · %s 用纯测光特征时，随机划分把 RMSE 压低了 %.1f%%（虚高 %.3f×）——'
                % (r['target'], r['rel_inflation_pct'], r['factor']))
            say('      因为同一颗星的测光完全相同，模型只是记住了这颗星的均值。')
        elif r['feature_set'] == 'obs':
            say('    · %s 用纯 S/N 特征时，两种划分差距 %.1f%%，说明观测级特征本身不构成恒星级泄漏。'
                % (r['target'], r['rel_inflation_pct']))

    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    with open(os.path.join(TABLES, 'leakage_report.txt'), 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
