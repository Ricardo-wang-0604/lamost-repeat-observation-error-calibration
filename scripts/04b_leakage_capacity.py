# -*- coding: utf-8 -*-
"""
04b_leakage_capacity.py —— 标签泄漏的**决定性**实验

为什么需要这个脚本
------------------
第一版实验（04_leakage.py）用 LightGBM 默认正则（min_child_samples=40）跑，
只测到 1-2% 的差异。这不是"泄漏不存在"，而是**实验设计把记忆这条路堵死了**：

  同一颗星通常只有 2 次观测，而 LightGBM 要求每个叶子至少 40 个样本，
  于是这颗星的样本必然与 38 个"别的星"混在同一叶子 → 模型无法单独记住它。

本脚本从三个维度把效应打开：

  维度一 · 模型容量
      min_child_samples ∈ {1, 10, 40}（见 CAPACITIES 常量）
      容量越大 → 越能做出"恒星专属"的叶子 → 泄漏越显著
      实测 10/12 个（子集 × 标签）组合随 min_child_samples 增大而倍数单调下降，
      例外为 HIP/teff 与 HIP/logg（HIP 是最小子集，噪声更大）。

  维度二 · 最近邻（决定性对照）
      KNN(k=1)。随机划分下，测试观测在训练集中存在**特征完全相同**的样本
      （同一颗星的另一次观测），1-NN 必然返回同一颗星的标签，
      残差退化为组内离散；分组划分下最近邻只能是别的星。
      这一条不依赖任何超参调校，最没有辩解空间。
      实测 11/12 个组合中 1-NN 的倍数最大，例外为 HIP/teff。

  维度三 · 复观率
      在全样本、repeat>=3 子集、以及高复观率的 HIP / TD 计划上分别做。
      泄漏只发生在复观样本上，复观率越高效应越明显。

产出
----
results/tables/leakage_capacity.csv       容量扫描 + 最近邻对照（全表）
results/tables/leakage_capacity_report.txt 人读报告

注：早期版本此处曾承诺产出 `leakage_knn.csv` 与 `leakage_subsets.csv`，
但代码从未写过这两个文件 —— 所有结果都在单一的 `leakage_capacity.csv` 里
（用 `model` / `param` 两列区分模型），已更正文档。
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 项目根按本文件位置向上两级解析（scripts/ → 项目根）。
# 不再硬编码绝对路径 —— 否则别人 clone 到别的目录跑不起来，
# 也无法把项目整体复制到临时目录做安全试跑。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import loader, leakage as lk  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LOG = []
TARGETS = ['teff', 'logg', 'feh']

# 计算预算控制
# -------------
# 全量 745 万行 × 4 子集 × 3 标签 × 4 模型 × 2 划分 会超过 2 小时。
# 泄漏效应由**分组结构**决定，不依赖样本绝对大小；因此对每个子集做
# **按恒星（gp_id）整组保留**的确定性下采样 —— 注意**不能按观测抽样**，
# 那会打散复观配对、把效应抹平（见 subsample() 的 docstring 与 README §4.3）。
MAX_N = 400_000
N_ESTIMATORS = 200
CAPACITIES = (1, 10, 40)
KNN_K = (1,)
SUBSET_NAMES = ('all', 'repeat>=3', 'HIP', 'TD')


def say(s=''):
    print(s, flush=True)
    LOG.append(str(s))


def lgbm(mcs):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=0.08,
                             num_leaves=255, min_child_samples=mcs,
                             subsample=0.9, subsample_freq=1,
                             colsample_bytree=0.9, random_state=42, n_jobs=-1,
                             verbose=-1)


def subsample(sub, seed=0, key='gp_id'):
    """
    按**恒星**下采样：随机选一批 gp_id，把它们的所有观测整组保留。

    为什么不能按观测下采样
    ----------------------
    一颗星通常只有 2 次观测。若对观测随机抽样（例如抽 5.4%），
    两次观测都被抽中的概率极低 —— 复观配对会被打散。实测按观测
    抽 40 万行后复观率从 1.381× 掉到 1.032×，精确命中率仅 4.8%，
    泄漏效应自然被抹平。这是第一版实验的**设计缺陷**。

    按恒星抽样则完整保留组内结构，复观率不变。
    """
    if len(sub) <= MAX_N:
        return sub
    uniq = sub[key].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    # 按随机抽到的星顺序累积，直到达到 MAX_N。
    # （注释早期写"按组大小从大到小累积"，但代码从未排序 —— 随机顺序更无偏，注释已更正。）
    sizes = sub.groupby(key, observed=True).size()
    order = sizes.reindex(uniq).fillna(0).to_numpy()
    cum = np.cumsum(order)
    k = int(np.searchsorted(cum, MAX_N) + 1)
    kept = set(uniq[:k].tolist())
    out = sub[sub[key].isin(kept)]
    return out


def knn(k):
    """
    最近邻回归。

    注意 n_jobs 必须为 1：
    sklearn 的 KNN 预测走 joblib，而即使声明 prefer="threads"，
    joblib 仍会构造 multiprocessing.SimpleQueue 传递回调，
    该队列底层是**命名管道** —— 在本项目的沙箱环境下会被拒绝：
        PermissionError: [WinError 5] 拒绝访问 (multiprocessing\\connection.py Pipe)
    LightGBM 用 OpenMP 线程所以不受影响，KNN 必须串行。
    """
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=k, n_jobs=1)


def row_fingerprint(X):
    """
    把每行特征压成一个不可变字节块，用于**快速判断两行是否完全相同**。

    这是本实验的关键量：同一颗星多次观测的测光特征在字节级别完全一致，
    因此随机划分下测试观测可能在训练集里存在"一模一样"的行 —— 这正是泄漏的来源。
    用 numpy 的 void 视图一次算完，比逐行 tuple() 快几个数量级。
    """
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    return Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()


def evaluate(X, y, g, tr, te, model, tag=''):
    model.fit(X[tr], y[tr])
    p = model.predict(X[te])
    rmse = float(np.sqrt(np.mean((y[te] - p) ** 2)))
    fp_tr = row_fingerprint(X[tr])
    fp_te = row_fingerprint(X[te])
    hit = int(np.isin(fp_te, fp_tr).sum())
    return {'tag': tag, 'rmse': rmse, 'n_test': int(len(te)),
            'n_test_exact_hit': hit,
            'frac_exact_hit': hit / max(len(te), 1)}


def run_pair(X, y, g, make, test_size=0.2, seed=42, tag=''):
    tr, te = lk.split_random(X, y, g, test_size, seed)
    a = evaluate(X, y, g, tr, te, make(), tag + '/random')
    tr, te = lk.split_grouped(g, test_size, seed)
    b = evaluate(X, y, g, tr, te, make(), tag + '/grouped')
    return a, b


def demo_subsampling(df, max_n=400_000, seed=0, key='gp_id'):
    """
    演示两种下采样方式的差别 —— 这是本项目最重要的方法论教训的可复现证明。

    纯统计计算（**不训练任何模型**，秒级），因此可以每次都跑：

      方案 A 按**观测**随机抽            → 复观配对被打散
      方案 B 按**恒星**整组保留（本项目） → 复观结构不变

    对每种方案报告：
      · 抽样后的复观率 = 行数 / 唯一恒星数
      · 随机划分下的"精确命中率" = 测试观测中，其**特征向量与某条训练样本完全相同**
        的比例（用 numpy void 视图做行级指纹）。这正是泄漏的直接来源。

    实测：按观测抽样会把复观率从 1.381× 打到约 1.03×，精确命中率降到约 5%，
    于是泄漏效应**完全检测不到**；按恒星抽样则保留 1.35×，命中率约 37%。

    见 README §4.3。
    """
    def fingerprint(X):
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        return Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()

    feat = [c for c in lk.STAR_LEVEL_FEATURES if c in df.columns]
    if df.empty or not feat:
        return
    rng = np.random.RandomState(seed)

    # --- 方案 A：按观测抽样 ---
    if len(df) > max_n:
        idx = np.sort(rng.choice(len(df), size=max_n, replace=False))
        A = df.iloc[idx]
    else:
        A = df
    # --- 方案 B：按恒星抽样 ---
    if len(df) > max_n:
        uniq = df[key].unique()
        rng.shuffle(uniq)
        sizes = df.groupby(key, observed=True).size()
        order = sizes.reindex(uniq).fillna(0).to_numpy()
        k = int(np.searchsorted(np.cumsum(order), max_n) + 1)
        B = df[df[key].isin(set(uniq[:k].tolist()))]
    else:
        B = df

    say('\n' + '=' * 86)
    say('  下采样方式对比（纯统计，不训练模型）—— README §4.3 的可复现证明')
    say('=' * 86)
    say('  %-22s %10s %10s %12s %14s'
        % ('方案', '行数', '唯一恒星', '复观率', '随机划分精确命中率'))
    for lab, sub in (('A 按观测抽样', A), ('B 按恒星整组保留', B)):
        n = len(sub)
        ng = sub[key].nunique()
        # 随机划分下测试集精确命中率
        sub2 = sub.dropna(subset=feat + ['teff'])
        if len(sub2) < 1000:
            say('  %-22s %10d %10d %12.3f %14s' % (lab, n, ng, n / max(ng, 1), '-'))
            continue
        X = sub2[feat].to_numpy(dtype=np.float64)
        m = len(X)
        n_te = int(round(m * 0.2))
        perm = rng.permutation(m)
        te, tr = perm[:n_te], perm[n_te:]
        hit = float(np.isin(fingerprint(X[te]), fingerprint(X[tr])).mean())
        say('  %-22s %10d %10d %12.3f %13.1f%%'
            % (lab, n, ng, n / max(ng, 1), 100 * hit))
    say('\n  按观测抽样把复观配对打散 ⇒ 精确命中率骤降 ⇒ 泄漏效应无法检出。')
    say('  这是本项目第一版实验失败的**根本原因**，不是"泄漏不存在"。')

    with open(os.path.join(TABLES, 'subsampling_demo.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))


def main():
    t0 = time.time()
    say('=' * 86)
    say('  标签泄漏的决定性实验：容量 / 最近邻 / 复观率')
    say('=' * 86)

    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    say('载入 %d 行' % len(df))
    if 'plan_prefix' not in df.columns:
        df['plan_prefix'] = (df['planid'].astype('string')
                             .str.extract(r'^([A-Za-z]+)', expand=False))
    df['plan_prefix'] = df['plan_prefix'].astype('string').fillna('UNK')

    # ---------------- 先验证下采样方式的影响（秒级，不训练模型）----------------
    # 这一步产出 README §4.3 引用的可复现证明：按观测抽样会打散复观配对。
    # `--demo-only` 只跑这一步就退出（无需等待 35 分钟的模型训练）。
    if '--skip-demo' not in sys.argv:
        try:
            demo_subsampling(df)
        except Exception as exc:  # noqa: BLE001
            say('[下采样对比跳过] %s' % exc)
    if '--demo-only' in sys.argv:
        return 0

    # ---------------- 子集定义 ----------------
    cnt = df.groupby('gp_id', observed=True).size()
    nrep = df['gp_id'].map(cnt).fillna(0).to_numpy()
    masks = {
        'all': np.ones(len(df), dtype=bool),
        'repeat>=3': nrep >= 3,
        'HIP': (df['plan_prefix'] == 'HIP').to_numpy(),
        'TD': (df['plan_prefix'] == 'TD').to_numpy(),
    }

    # ---------------- 维度一/三：容量 × 子集 ----------------
    rows = []
    for sname in SUBSET_NAMES:
        mask = masks[sname]
        sub = subsample(df[mask])
        if len(sub) < 2000:
            say('\n[跳过 %s] 样本不足 (%d)' % (sname, len(sub)))
            continue
        say('\n[子集 %s] 下采样后 %d 行（原始 %d）' % (sname, len(sub), int(mask.sum())))
        for tgt in TARGETS:
            try:
                X, y, g, cols, _ = lk.build_matrix(sub, feature_set='photometry',
                                                   target=tgt)
            except Exception as exc:  # noqa: BLE001
                say('[跳过 %s/%s] %s' % (sname, tgt, exc))
                continue
            if len(y) < 2000:
                continue
            say('\n' + '-' * 86)
            say('子集=%-9s 标签=%-5s 样本=%d 星=%d 复观率=%.3f×'
                % (sname, tgt, len(y), len(np.unique(g)), len(y) / len(np.unique(g))))
            say('  %-22s %12s %12s %9s %12s' %
                ('模型', 'RMSE随机', 'RMSE分组', '倍数', '随机/精确命中率'))
            for mcs in CAPACITIES:
                a, b = run_pair(X, y, g, lambda m=mcs: lgbm(m), tag='lgbm%d' % mcs)
                fac = b['rmse'] / a['rmse'] if a['rmse'] else np.nan
                say('  %-22s %12.4f %12.4f %8.3fx %11.1f%%'
                    % ('lgbm mcs=%d' % mcs, a['rmse'], b['rmse'], fac,
                       100 * a['frac_exact_hit']))
                rows.append({'subset': sname, 'target': tgt, 'model': 'lgbm',
                             'param': 'min_child_samples=%d' % mcs,
                             'rmse_random': a['rmse'], 'rmse_grouped': b['rmse'],
                             'factor': fac,
                             'frac_exact_hit_random': a['frac_exact_hit'],
                             'frac_exact_hit_grouped': b['frac_exact_hit'],
                             'n_samples': len(y), 'n_groups': int(len(np.unique(g)))})
            # KNN 对照
            for k in KNN_K:
                a, b = run_pair(X, y, g, lambda kk=k: knn(kk), tag='knn%d' % k)
                fac = b['rmse'] / a['rmse'] if a['rmse'] else np.nan
                say('  %-22s %12.4f %12.4f %8.3fx %11.1f%%'
                    % ('knn k=%d' % k, a['rmse'], b['rmse'], fac,
                       100 * a['frac_exact_hit']))
                rows.append({'subset': sname, 'target': tgt, 'model': 'knn',
                             'param': 'k=%d' % k,
                             'rmse_random': a['rmse'], 'rmse_grouped': b['rmse'],
                             'factor': fac,
                             'frac_exact_hit_random': a['frac_exact_hit'],
                             'frac_exact_hit_grouped': b['frac_exact_hit'],
                             'n_samples': len(y), 'n_groups': int(len(np.unique(g)))})

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(TABLES, 'leakage_capacity.csv'), index=False,
             encoding='utf-8-sig')

    # ---------------- 汇总 ----------------
    say('\n' + '=' * 86)
    say('  汇总：泄漏倍数 factor = RMSE_grouped / RMSE_random')
    say('=' * 86)
    if len(R):
        piv = R.pivot_table(index=['subset', 'target'], columns=['model', 'param'],
                            values='factor')
        say(piv.to_string(float_format=lambda x: '%6.3f' % x))

        say('\n  关键读数：')
        # 1-NN 在全样本上的倍数
        k1 = R[(R['model'] == 'knn') & (R['param'] == 'k=1') & (R['subset'] == 'all')]
        for _, r in k1.iterrows():
            say('    · 1-近邻 / 全样本 / %-5s : %.3f×  (随机划分测试集中有 %.1f%% 的观测'
                '在训练集里存在特征完全相同的样本)' %
                (r['target'], r['factor'], 100 * r['frac_exact_hit_random']))
        # 容量趋势
        for sub in R['subset'].unique():
            s = R[(R['subset'] == sub) & (R['model'] == 'lgbm') & (R['target'] == 'teff')]
            if s.empty:
                continue
            s = s.sort_values('param')
            say('    · lgbm / %-9s / teff : 倍数 由 %.3f× (mcs=%s) 变到 %.3f× (mcs=%s)'
                % (sub, s['factor'].iloc[0],
                   s['param'].iloc[0].split('=')[1],
                   s['factor'].iloc[-1],
                   s['param'].iloc[-1].split('=')[1]))

    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    with open(os.path.join(TABLES, 'leakage_capacity_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
