# -*- coding: utf-8 -*-
"""
ml/predict.py —— 单场比赛特征构造器
====================================
输入两位选手名 + 比赛日期(可选的赛事级别/主客场), 复用 data_pipeline.py 的
同一套公式, 算出 8 个特征, 供 app/main.py 喂给模型预测胜率。

特征口径与训练时 data_pipeline.build_features 完全一致:
  1. 解析选手名 -> 排名表标准名
  2. 赛前排名(as-of 查询) -> points_share / rank_diff / points_diff
  3. 扫历史战绩(严格早于比赛日期) -> form5_1 / form5_2 / h2h_winrate
  4. 可选输入 -> tier_level / home_adv
"""
import os
import pandas as pd

from ml import data_pipeline as dp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MS_PATH = os.path.join(_ROOT, "data", "raw", "ms.csv")
_RK_PATH = os.path.join(_ROOT, "data", "raw", "bwf_historic_rankings.csv")

# 8 个特征(与 train.py 的 V2 顺序一致)
FEATURES = ["points_share", "h2h_winrate", "form5_1", "form5_2",
            "rank_diff", "points_diff", "tier_level", "home_adv"]

# 未提供赛事级别时的默认级别分(中等级别)
DEFAULT_TIER_LEVEL = 5

_cache = None


def load_data():
    """懒加载 + 缓存: 返回 (ms_mapped, rank_lookup, rk_names)。

    ms_mapped: 比赛表(已映射成排名表标准名、去掉资格赛、按日期排序), 列含
               date / p1 / p2 / win1(team1 是否赢)
    rank_lookup: 每个标准名 -> 按日期升序的 (日期, 名次, 积分) 列表
    rk_names: 排名表标准名列表
    """
    global _cache
    if _cache is not None:
        return _cache

    ms = pd.read_csv(_MS_PATH, low_memory=False)
    rk = pd.read_csv(_RK_PATH, low_memory=False)

    ms["date"] = pd.to_datetime(ms["date"])
    rk["date"] = pd.to_datetime(rk["date"])

    # 和 load_and_filter 完全一致: 2008-2022 + 男单 + 去名字前导空格
    ms = ms[(ms["date"] >= dp.START) & (ms["date"] <= dp.END)].copy()
    rk = rk[(rk["date"] >= dp.START) & (rk["date"] <= dp.END)].copy()
    rk = rk[rk["draw"] == "MS"].copy()
    rk["name"] = rk["name"].astype(str).str.strip()

    rk_names = sorted(rk["name"].unique())
    name_map = dp.build_name_map(set(ms["team1"]) | set(ms["team2"]), rk_names)

    # 比赛表映射成标准名, 只留双方都能映射到排名的 + 非资格赛
    m = ms.copy()
    m["p1"] = m["team1"].map(name_map)
    m["p2"] = m["team2"].map(name_map)
    m = m[m["p1"].notna() & m["p2"].notna()]
    m = m[~m["round"].astype(str).str.contains("Qual", case=False, na=False)]
    m["win1"] = m["winner"] == 1
    m = m.sort_values("date").reset_index(drop=True)

    lookup = dp.build_rank_lookup(rk)

    _cache = (m, lookup, rk_names)
    return _cache


def resolve_player(query, rk_names):
    """把用户输入的(部分)名字解析成排名表标准名。

    返回 (标准名 或 None, 候选列表)。候选非空表示存在歧义。
    """
    qt = dp.tokens(query)
    if not qt:
        return None, []

    # 输入 token 是某标准名 token 的子集(解决 "Axelsen" -> "VIKTOR AXELSEN")
    candidates = sorted({n for n in rk_names if qt.issubset(dp.tokens(n))},
                        key=lambda n: len(dp.tokens(n)))
    if len(candidates) == 1:
        return candidates[0], []

    # 多个候选时, 先看是否有 token 完全相等的那一个
    exact = [n for n in candidates if dp.tokens(n) == qt]
    if len(exact) == 1:
        return exact[0], []

    return None, candidates


def _h2h_form(ms_mapped, p1, p2, d):
    """扫描严格早于 d 的比赛, 返回 (hist1, hist2, w1, w2)。

    hist1/hist2: 双方各自的胜场序列(按时间), 用于近5场胜率。
    w1/w2: 双方历史交手中各自的胜场数。
    """
    hist1, hist2 = [], []
    w1 = w2 = 0

    rel = ms_mapped[ms_mapped["date"] < d]
    rel = rel[rel["p1"].isin([p1, p2]) | rel["p2"].isin([p1, p2])]

    for _, r in rel.iterrows():
        a, b, win1 = r["p1"], r["p2"], bool(r["win1"])
        if a == p1:
            hist1.append(win1)
            if b == p2:
                hist2.append(not win1)
                w1 += 1 if win1 else 0
                w2 += 0 if win1 else 1
        elif b == p1:
            hist1.append(not win1)
            if a == p2:
                hist2.append(win1)
                w2 += 1 if win1 else 0
                w1 += 0 if win1 else 1
        elif a == p2:
            hist2.append(win1)
        elif b == p2:
            hist2.append(not win1)

    return hist1, hist2, w1, w2


def build_match_features(player1, player2, match_date, tier=None, home_adv=None):
    """构造一场比赛的 8 个特征。

    参数:
      player1 / player2: 选手名(可部分匹配, 如 "Axelsen")
      match_date:        比赛日期(字符串或 date/datetime)
      tier:              赛事级别字符串(可选, 如 "HSBC BWF World Tour Super 1000")
      home_adv:          主客场(可选, -1/0/1; 1=player1 主场, -1=player2 主场)

    返回:
      {"player1": 标准名, "player2": 标准名, "features": {8个特征}}

    找不到选手 / 名字歧义 / 查不到排名时抛 ValueError。
    """
    ms_mapped, lookup, rk_names = load_data()
    d = pd.to_datetime(match_date)

    p1, cand1 = resolve_player(player1, rk_names)
    p2, cand2 = resolve_player(player2, rk_names)
    if p1 is None:
        cand = ", ".join(cand1) if cand1 else "无候选"
        raise ValueError(f"找不到选手 '{player1}'(候选: {cand})")
    if p2 is None:
        cand = ", ".join(cand2) if cand2 else "无候选"
        raise ValueError(f"找不到选手 '{player2}'(候选: {cand})")
    if p1 == p2:
        raise ValueError(f"两名选手解析成了同一个人: {p1}")

    # ---- 赛前排名(as-of 查询) ----
    l1 = dp.latest_rank(p1, d, lookup)
    l2 = dp.latest_rank(p2, d, lookup)
    if l1 is None or l2 is None:
        who = p1 if l1 is None else p2
        raise ValueError(f"比赛日期 {match_date} 之前查不到 '{who}' 的排名")
    _, rank1, pts1 = l1
    _, rank2, pts2 = l2

    points_share = pts1 / (pts1 + pts2)
    rank_diff = rank1 - rank2
    points_diff = pts1 - pts2

    # ---- 历史战绩(严格早于比赛日期, 防泄漏) ----
    hist1, hist2, w1, w2 = _h2h_form(ms_mapped, p1, p2, d)
    form5_1 = dp.win_rate_last_n(hist1, 5)
    form5_2 = dp.win_rate_last_n(hist2, 5)
    n = w1 + w2
    h2h_winrate = (w1 + 1) / (n + 2)          # 贝叶斯平滑, 没交过手=0.5

    # ---- 可选输入 ----
    tier_val = dp.tier_level(tier) if tier else DEFAULT_TIER_LEVEL
    home_val = int(home_adv) if home_adv is not None else 0

    features = {
        "points_share": points_share,
        "h2h_winrate": h2h_winrate,
        "form5_1": form5_1,
        "form5_2": form5_2,
        "rank_diff": rank_diff,
        "points_diff": points_diff,
        "tier_level": tier_val,
        "home_adv": home_val,
    }
    return {"player1": p1, "player2": p2, "features": features}
