# -*- coding: utf-8 -*-
"""
run_extract.py —— LAMOST DR10 stellar 表全量抽取（游标分页，可断点续跑）

字段集含 planid / spid / fiberid：
  * planid 用于区分观测计划（HIP 标准星 / HD 巡天 / KP 开普勒 / TD 时域 / KII）
  * spid + fiberid 用于把目录行与 FITS 光谱文件对上
    （日包路径形如 20111024/F5902/spec-55859-F5902_sp01-050.fits.gz）

实测约束（LAMOST DR10 v2.0 TAP, 2026-09-26）
  * 只输出 VOTable；FORMAT=csv/json 被忽略
  * 不支持子查询、不支持 ADQL 几何函数
  * `top N` 无 `order by` 时顺序不确定 → 必须显式排序才可复现
  * obsid 唯一，可作游标分页键；深分页耗时恒定 ~18.5s/20000 行
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tap as lamost  # noqa: E402

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
PAGES = os.path.join(ROOT, 'data', 'raw', 'stellar_pages')
PROG = os.path.join(ROOT, 'data', 'raw', 'progress.json')
PAGE_SIZE = 20000
COLS = lamost.STELLAR_COLS

os.makedirs(PAGES, exist_ok=True)


def load_prog():
    if os.path.exists(PROG):
        with open(PROG, encoding='utf-8') as f:
            return json.load(f)
    return {'cursor': 101000, 'page': 0, 'rows': 0, 'done': False}


def save_prog(p):
    tmp = PROG + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(p, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PROG)


def main():
    prog = load_prog()
    print('=' * 72)
    print('  LAMOST stellar 全量抽取')
    print('  cursor=%d  page=%d  rows=%d  done=%s'
          % (prog['cursor'], prog['page'], prog['rows'], prog['done']))
    print('=' * 72)

    t0 = time.time()
    while not prog['done']:
        cur = prog['cursor']
        try:
            fields, rows = lamost.cursor_page(cur, PAGE_SIZE, COLS)
        except Exception as exc:  # noqa: BLE001
            print('  !! 第 %d 页失败：%s' % (prog['page'], str(exc)[:200]))
            save_prog(prog)
            return 2

        if not rows:
            # `obsid > cur order by obsid` 是无界查询：空 ⇔ 已无更大 obsid。
            # 注意不能靠翻倍游标"跳过"——obsid 区间稀疏只发生在有界查询里。
            prog['done'] = True
            save_prog(prog)
            print('  无更大 obsid → 抽取完成')
            break

        path = os.path.join(PAGES, 'page_%05d.csv.gz' % prog['page'])
        with gzip.open(path, 'wt', encoding='utf-8-sig', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(fields)
            w.writerows(rows)

        prog['cursor'] = int(rows[-1][0])
        prog['page'] += 1
        prog['rows'] += len(rows)
        save_prog(prog)

        el = time.time() - t0
        if prog['page'] % 5 == 0 or prog['page'] <= 3:
            print('  页 %-4d 累计 %9d 行 游标=%-12d 用时 %6.1f 分 速率 %.0f 行/分'
                  % (prog['page'], prog['rows'], prog['cursor'],
                     el / 60.0, prog['rows'] / max(el, 1) * 60))

        if len(rows) < PAGE_SIZE:
            prog['done'] = True
            save_prog(prog)
            print('  末页不足 %d 行 → 抽取完成' % PAGE_SIZE)
            break
        time.sleep(0.2)

    save_prog(prog)
    print('=' * 72)
    print('  结束: 页数=%d 行数=%d 用时=%.1f 分'
          % (prog['page'], prog['rows'], (time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
