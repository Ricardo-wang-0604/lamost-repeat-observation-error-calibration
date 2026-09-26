# -*- coding: utf-8 -*-
"""
run_extract_seg.py —— LAMOST stellar 表**分片并行**抽取

为什么要分片
------------
单进程游标分页实测 25.8 秒/页，373 页需约 2.4 小时。
obsid 空间是 [101001, 1018116242]，把它切成 N 段并发抽取可近线性提速。

用法
----
    python scripts/run_extract_seg.py <worker_id> <n_workers>

每个 worker 只负责自己的 obsid 区间 [lo, hi)，输出独立的
data/raw/seg{worker}/page_*.csv.gz 与 progress.json，互不干扰，
可各自断点续跑。

worker 0 会自动接管旧的 data/raw/stellar_pages 里已完成的部分
（若其游标落在自己区间内），避免重复劳动。
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tap as lamost  # noqa: E402

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
RAW = os.path.join(ROOT, 'data', 'raw')
OBSID_MIN = 101001
OBSID_MAX = 1018116242
PAGE_SIZE = 20000
COLS = lamost.STELLAR_COLS


def segment(w, n):
    span = (OBSID_MAX - OBSID_MIN) // n
    lo = OBSID_MIN + w * span
    hi = OBSID_MAX if w == n - 1 else OBSID_MIN + (w + 1) * span
    return lo, hi


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    w = int(sys.argv[1])
    n = int(sys.argv[2])
    lo, hi = segment(w, n)

    seg_dir = os.path.join(RAW, 'seg%d' % w)
    os.makedirs(seg_dir, exist_ok=True)
    prog_path = os.path.join(seg_dir, 'progress.json')

    # 续跑
    if os.path.exists(prog_path):
        with open(prog_path, encoding='utf-8') as f:
            prog = json.load(f)
        if prog.get('done'):
            print('[worker %d] 已完成，跳过' % w)
            return 0
        cursor = prog['cursor']
        page = prog['page']
        rows_total = prog['rows']
    else:
        cursor = lo
        page = 0
        rows_total = 0
        # worker 0 承接旧的顺序抽取成果
        old_prog = os.path.join(RAW, 'progress.json')
        old_pages = os.path.join(RAW, 'stellar_pages')
        if w == 0 and os.path.exists(old_prog):
            with open(old_prog, encoding='utf-8') as f:
                op = json.load(f)
            old_cursor = op.get('cursor', 0)
            if lo <= old_cursor < hi:
                files = sorted(glob.glob(os.path.join(old_pages, 'page_*.csv.gz')))
                for i, p in enumerate(files):
                    shutil.copy2(p, os.path.join(seg_dir, 'page_%05d.csv.gz' % i))
                cursor = old_cursor
                page = len(files)
                rows_total = op.get('rows', 0)
                print('[worker 0] 承接旧进度: %d 页, cursor=%d' % (page, cursor))

    print('=' * 70)
    print('[worker %d/%d] obsid ∈ [%d, %d)  起始 cursor=%d  page=%d  rows=%d'
          % (w, n, lo, hi, cursor, page, rows_total))
    print('=' * 70, flush=True)

    t0 = time.time()
    while cursor < hi:
        adql = ('select top %d %s from stellar where obsid > %d and obsid <= %d '
                'order by obsid' % (PAGE_SIZE, COLS, cursor, hi))
        ok = False
        for attempt in range(4):
            try:
                fields, rows = lamost.query(adql, timeout=900, attempts=1)
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print('[worker %d] 第 %d 页尝试 %d 失败: %s'
                      % (w, page, attempt + 1, str(exc)[:150]), flush=True)
                time.sleep(5 + 5 * attempt)
        if not ok:
            print('[worker %d] 连续失败，保存进度退出' % w, flush=True)
            break

        if not rows:
            print('[worker %d] 区间内无更多行 → 完成' % w, flush=True)
            break

        path = os.path.join(seg_dir, 'page_%05d.csv.gz' % page)
        with gzip.open(path, 'wt', encoding='utf-8-sig', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(fields)
            wr.writerows(rows)

        cursor = int(rows[-1][0])
        page += 1
        rows_total += len(rows)
        with open(prog_path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump({'cursor': cursor, 'page': page, 'rows': rows_total,
                       'lo': lo, 'hi': hi, 'done': False}, f, indent=1)
        os.replace(prog_path + '.tmp', prog_path)

        if page % 5 == 0:
            el = time.time() - t0
            print('[worker %d] 页 %-4d 本段累计 %8d 行 cursor=%-12d 用时 %5.1f 分'
                  % (w, page, rows_total, cursor, el / 60.0), flush=True)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.15)

    with open(prog_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump({'cursor': cursor, 'page': page, 'rows': rows_total,
                   'lo': lo, 'hi': hi, 'done': True}, f, indent=1)
    os.replace(prog_path + '.tmp', prog_path)
    print('[worker %d] 结束: 页数=%d 行数=%d 用时=%.1f 分'
          % (w, page, rows_total, (time.time() - t0) / 60.0), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
