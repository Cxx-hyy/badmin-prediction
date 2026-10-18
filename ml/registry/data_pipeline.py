# -*- coding: utf-8 -*-
"""
数据管道：从原始比赛结果数据 (v1) 构造训练特征数据 (v2)

项目：羽毛球男单比赛胜负预测
v1 (match_stats_v1.csv)：原始比赛结果，含文本字段与缺失值
v2 (match_stats_v2.csv)：清洗 + 特征工程后的训练数据

特征设计（严格避免数据泄露）：
- 所有特征均按比赛时间顺序、在"比赛开始前"基于选手历史计算，
  计算完成后才用本场结果更新统计量。
- 特征：
  * margin-adjusted Elo（大胜多涨分，K 随局点差放大）
  * 历史胜率、近期状态(近 N 场)、交手记录(H2H)、参赛经验
  * 历史场均局点差 (point margin)、历史局胜率（来自比分 score）
  * 主客场
- 标签 label：team1 获胜为 1，否则为 0。
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Elo 参数
ELO_BASE = 1500.0
ELO_K = 32.0
# 近期状态窗口
RECENT_N = 10
# 双方选手在本场前至少要有多少场历史比赛，才纳入训练集
MIN_MATCHES = 5
# 无历史信息时的中立胜率
NEUTRAL = 0.5
# 大胜时 K 的放大系数（局点差越大，胜负信息越可信）
MARGIN_SCALE = 0.5


def load_raw(path=None):
    """加载 v1 原始数据。"""
    path = path or (DATA_DIR / "match_stats_v1.csv")
    return pd.read_csv(path, low_memory=False)


def clean(df):
    """清洗：解析日期、过滤无效行、补全主客场、按时间排序。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team1", "team2", "winner"])
    df["winner"] = df["winner"].astype(int)
    df = df[df["winner"].isin([1, 2])]
    df["team1_at_home"] = df["team1_at_home"].fillna(False).astype(bool)
    df["team2_at_home"] = df["team2_at_home"].fillna(False).astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def parse_score(score):
    """解析比分（如 '21-15 17-21 21-16'）。
    返回 (team1总得分, team2总得分, team1赢局数, team2赢局数, 是否有效)。"""
    p1p = p2p = p1g = p2g = 0
    ok = False
    for game in str(score).split():
        parts = game.split("-")
        if len(parts) != 2:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        p1p += a
        p2p += b
        if a > b:
            p1g += 1
        elif b > a:
            p2g += 1
        ok = True
    return p1p, p2p, p1g, p2g, ok


def _expected_score(elo_a, elo_b):
    """Elo 期望胜率：a 战胜 b 的概率。"""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def build_features(df, min_matches=MIN_MATCHES):
    """单遍按时间顺序遍历比赛，构造赛前特征（无数据泄露）。"""
    elo, wins, match_count, recent = {}, {}, {}, {}
    games, pts_margin, games_won = {}, {}, {}
    h2h = {}
    rows = []

    for rec in df.itertuples(index=False):
        p1, p2, winner = rec.team1, rec.team2, int(rec.winner)
        p1p, p2p, p1g, p2g, score_ok = parse_score(rec.score)

        # ---- 比赛前：读取双方历史状态 ----
        e1, e2 = elo.get(p1, ELO_BASE), elo.get(p2, ELO_BASE)
        m1, m2 = match_count.get(p1, 0), match_count.get(p2, 0)
        wr1 = wins.get(p1, 0) / m1 if m1 else NEUTRAL
        wr2 = wins.get(p2, 0) / m2 if m2 else NEUTRAL
        form1 = float(np.mean(recent[p1])) if recent.get(p1) else NEUTRAL
        form2 = float(np.mean(recent[p2])) if recent.get(p2) else NEUTRAL
        margin1 = pts_margin[p1] / games[p1] if games.get(p1) else 0.0
        margin2 = pts_margin[p2] / games[p2] if games.get(p2) else 0.0
        gwr1 = games_won[p1] / games[p1] if games.get(p1) else NEUTRAL
        gwr2 = games_won[p2] / games[p2] if games.get(p2) else NEUTRAL

        key = tuple(sorted([p1, p2]))
        pair = h2h.get(key)
        if pair:
            h2h_matches = pair["total"]
            h2h_wr1 = pair["wins"].get(p1, 0) / pair["total"]
        else:
            h2h_matches, h2h_wr1 = 0, NEUTRAL

        label = 1 if winner == 1 else 0

        if m1 >= min_matches and m2 >= min_matches:
            rows.append({
                "date": rec.date.strftime("%Y-%m-%d"),
                "team1": p1, "team2": p2,
                # 信息列（便于解释/展示，不全用于训练）
                "elo_1": round(e1, 2), "elo_2": round(e2, 2),
                "win_rate_1": round(wr1, 4), "win_rate_2": round(wr2, 4),
                "matches_1": m1, "matches_2": m2,
                # 建模特征
                "elo_diff": round(e1 - e2, 2),
                "win_rate_diff": round(wr1 - wr2, 4),
                "recent_form_1": round(form1, 4),
                "recent_form_2": round(form2, 4),
                "h2h_matches": h2h_matches,
                "h2h_win_rate_1": round(h2h_wr1, 4),
                "point_margin_1": round(margin1, 3),
                "point_margin_2": round(margin2, 3),
                "point_margin_diff": round(margin1 - margin2, 3),
                "game_win_rate_1": round(gwr1, 4),
                "game_win_rate_2": round(gwr2, 4),
                "at_home_1": int(bool(rec.team1_at_home)),
                "at_home_2": int(bool(rec.team2_at_home)),
                "label": label,
            })

        # ---- 比赛后：用本场结果更新统计量 ----
        win_p, lose_p = (p1, p2) if winner == 1 else (p2, p1)
        e_w, e_l = elo.get(win_p, ELO_BASE), elo.get(lose_p, ELO_BASE)
        exp_w = _expected_score(e_w, e_l)
        if score_ok:
            w_pts = p1p if winner == 1 else p2p
            l_pts = p2p if winner == 1 else p1p
            k_w = ELO_K * (1.0 + max(w_pts - l_pts, 1) / 21.0 * MARGIN_SCALE)
        else:
            k_w = ELO_K
        elo[win_p] = e_w + k_w * (1.0 - exp_w)
        elo[lose_p] = e_l - k_w * (1.0 - exp_w)

        wins[win_p] = wins.get(win_p, 0) + 1
        match_count[win_p] = match_count.get(win_p, 0) + 1
        match_count[lose_p] = match_count.get(lose_p, 0) + 1
        recent.setdefault(win_p, []).append(1)
        recent.setdefault(lose_p, []).append(0)
        recent[win_p] = recent[win_p][-RECENT_N:]
        recent[lose_p] = recent[lose_p][-RECENT_N:]

        if score_ok:
            total_games = p1g + p2g
            for p, pts, opp, gw in [(p1, p1p, p2p, p1g), (p2, p2p, p1p, p2g)]:
                games[p] = games.get(p, 0) + total_games
                pts_margin[p] = pts_margin.get(p, 0) + (pts - opp)
                games_won[p] = games_won.get(p, 0) + gw

        pair = h2h.setdefault(key, {"total": 0, "wins": {}})
        pair["total"] += 1
        pair["wins"][win_p] = pair["wins"].get(win_p, 0) + 1

    return pd.DataFrame(rows)


def make_dataset(out_path=None):
    """完整管道：v1 -> 清洗 -> 特征 -> v2，并保存为 CSV。"""
    cleaned = clean(load_raw())
    features = build_features(cleaned)
    out_path = out_path or (DATA_DIR / "match_stats_v2.csv")
    features.to_csv(out_path, index=False)
    return features


# baseline 仅使用的朴素特征（单一 Elo 信号 + 主场）
BASELINE_FEATURE_COLUMNS = ["elo_diff", "at_home_1", "at_home_2"]
# improved 使用的全部工程特征
FEATURE_COLUMNS = [
    "elo_diff", "win_rate_diff",
    "recent_form_1", "recent_form_2",
    "h2h_matches", "h2h_win_rate_1",
    "point_margin_1", "point_margin_2", "point_margin_diff",
    "game_win_rate_1", "game_win_rate_2",
    "at_home_1", "at_home_2",
]
TARGET_COLUMN = "label"


if __name__ == "__main__":
    v2 = make_dataset()
    print("v2 已生成：%s 行，%d 列" % (len(v2), v2.shape[1]))
    print("标签分布:\n", v2[TARGET_COLUMN].value_counts())
