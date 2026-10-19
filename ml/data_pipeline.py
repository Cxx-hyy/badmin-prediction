# -*- coding: utf-8 -*-
"""
02_data_pipeline.py —— 第二步: 数据清洗 + 特征工程(本项目核心)
================================================================
作用: 把两份"原始数据"变成一张"特征表"(每场比赛一行, 含特征 X 和标签 y)。
这张特征表是 03_train.py 训练模型的输入。

处理分 4 步(每个函数一步):
  1. load_and_filter()   读取 + 过滤 2008-2022 + 只留男单
  2. build_name_map()    把比赛表人名和排名表人名对齐
  3. build_rank_lookup() 建"每个选手 -> 按时间排序的排名历史"查询表
  4. build_features()    逐场比赛算特征(积分占比/交手胜率/近5场胜率...)

核心红线(务必理解, 这是整个项目的正确性所在):
  - 只用"赛前"信息: 排名取赛前最近一期; 交手/近况只用该场之前的战绩。
  - 按日期排序后逐场扫描, 保证"算第 N 场时, 只看得到前 N-1 场"。

【数据质量提醒】排名表里的 delta_rank(名次变化) 和 tournaments_played(参赛数)
两个字段, 在 1990-2022 全是空值(只有 2023-2024 有数据)。所以本项目(2008-2022)
不把它们当特征, 改用"赛事级别 tier"和"主客场 home_adv"这两个真正可用的字段。

运行:  python 02_data_pipeline.py
产出:  processed/features.csv
"""
import difflib
import os
import re
import sys

import pandas as pd

# 让 stdout 以 UTF-8 输出(Windows 控制台默认 GBK, 打印中文/特殊字符可能报错)
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(encoding="utf-8", errors="replace")

START = "2008-01-01"
END = "2022-12-31"


# ---------- 第 1 步: 读取 + 过滤 ----------
def load_and_filter():
    """读取两份 CSV, 只留 2008-2022 的男单数据。"""
    ms = pd.read_csv("data/raw/ms.csv", low_memory=False)
    rk = pd.read_csv("data/raw/bwf_historic_rankings.csv", low_memory=False)

    # 统一转成日期类型, 才能比较大小、做过滤
    ms["date"] = pd.to_datetime(ms["date"])
    rk["date"] = pd.to_datetime(rk["date"])

    # 只要 2008-2022(你要的时间范围)
    ms = ms[(ms["date"] >= START) & (ms["date"] <= END)].copy()
    rk = rk[(rk["date"] >= START) & (rk["date"] <= END)].copy()

    # 排名表只留男子单打
    rk = rk[rk["draw"] == "MS"].copy()

    # 排名表的人名前面有个多余空格, 去掉
    rk["name"] = rk["name"].str.strip()

    return ms, rk


# ---------- 第 2 步: 人名对齐 ----------
def tokens(s):
    """把名字转成"小写字母集合", 让大小写/单词顺序/标点不再影响比较。"""
    return set(re.sub(r"[^a-z\s]", " ", str(s).lower()).split())


def build_name_map(ms_players, rk_names):
    """
    建立映射: 比赛表人名 -> 排名表标准名。

    为什么不能直接按名字 join? 因为两份表格式不同:
      比赛表  "KHRISHNAN Yogendran"  (姓大写 + 名)
      排名表  "ANTHONY SINISUKA GINTING"  (全大写 + 可能带中名)

    三级策略:
      1) token 集合完全相等
      2) 子集包含 —— 解决"中名缺失", 如 "Anthony Ginting" 是 "ANTHONY SINISUKA GINTING" 的子集
      3) 模糊匹配 —— 解决拼写变体
    """
    rk_names = [str(n).strip() for n in rk_names if pd.notna(n)]
    rk_tok = [(n, tokens(n)) for n in rk_names]

    def match(p):
        pt = tokens(p)
        if len(pt) < 2:              # 单 token 名字太容易误配, 跳过
            return None
        # 1) 完全相等
        exact = [n for n, rt in rk_tok if pt == rt]
        if len(exact) == 1:
            return exact[0]
        # 2) 子集包含, 取 token 数最少的超集(最接近的那个)
        supers = [n for n, rt in rk_tok if pt.issubset(rt)]
        if supers:
            supers.sort(key=lambda n: len(tokens(n)))
            return supers[0]
        # 3) 模糊匹配(用排序后的 token 列表, 对顺序不敏感)
        best, best_sc = None, 0.0
        for n, _ in rk_tok:
            sc = difflib.SequenceMatcher(None, sorted(pt), sorted(tokens(n))).ratio()
            if sc > best_sc:
                best_sc, best = sc, n
        return best if best_sc >= 0.92 else None

    return {p: match(p) for p in ms_players}


# ---------- 第 3 步: 排名查询表 ----------
def build_rank_lookup(rk):
    """每个标准名 -> 按日期升序的 (日期, 名次, 积分) 列表。"""
    lookup = {}
    for name, g in rk.sort_values(["name", "date"]).groupby("name"):
        lookup[name] = list(zip(g["date"], g["rank"], g["points"], strict=True))
    return lookup


def latest_rank(name, match_date, lookup):
    """
    as-of 查询: 取 match_date 当天及之前最近一期排名。
    绝不取比赛之后的排名(那会数据泄漏)。
    """
    if name is None or name not in lookup:
        return None
    best = None
    for d, rank, pts in lookup[name]:
        if d <= match_date:
            best = (d, rank, pts)
        else:
            break
    return best


# ---------- 第 4 步: 造特征 ----------
def win_rate_last_n(hist, n):
    """最近 n 场胜率; 没打过取 0.5(中性先验)。"""
    recent = hist[-n:]                      # 取最后 n 场(即最近 n 场)
    return (sum(recent) / len(recent)) if recent else 0.5


def to_bool(v):
    """把 'True'/'False' 字符串或布尔值统一转成布尔。"""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def tier_level(t):
    """把赛事级别字符串映射成一个"级别分"(越高越重要)。"""
    s = str(t)
    if "Grade 1" in s: 
        return 8     # 世锦赛/奥运(最高)
    if "Finals" in s: 
        return 7      # 年终总决赛
    if "1000" in s: 
        return 7        # Super 1000
    if "750" in s: 
        return 6
    if "Premier" in s: 
        return 6     # 老顶级赛
    if "500" in s: 
        return 5
    if "Superseries" in s: 
        return 5
    if "300" in s: 
        return 4
    if "Gold" in s: 
        return 4        # Grand Prix Gold
    if "100" in s: 
        return 3
    if "Grand Prix" in s: 
        return 3
    return 2


def build_features(ms, rk):
    """把每场比赛变成一行特征, 返回 DataFrame。"""
    # 2.1 名字对齐
    players = set(ms["team1"]) | set(ms["team2"])
    name_map = build_name_map(players, rk["name"].unique())
    ms["p1"] = ms["team1"].map(name_map)
    ms["p2"] = ms["team2"].map(name_map)

    # 2.2 只保留"双方都进过前100"且"主赛程"的比赛
    #     (资格赛 Q Qual. 的选手大多没排名, 预测精英比赛时通常去掉)
    ms = ms[ms["p1"].notna() & ms["p2"].notna()]
    ms = ms[~ms["round"].astype(str).str.contains("Qual", case=False, na=False)]

    # 2.3 按日期排序 —— 关键! 保证流式扫描时"只看得到过去"
    ms = ms.sort_values("date").reset_index(drop=True)

    lookup = build_rank_lookup(rk)
    hist = {}    # 选手 -> [是否赢(按时间顺序)]
    h2h = {}     # (p1,p2)排序键 -> [先者胜场, 后者胜场]
    rows = []

    for _, r in ms.iterrows():
        d, p1, p2 = r["date"], r["p1"], r["p2"]
        win1 = r["winner"] == 1                       # team1 是否赢(这就是标签)

        l1 = latest_rank(p1, d, lookup)               # 双方"赛前"排名
        l2 = latest_rank(p2, d, lookup)
        if l1 is None or l2 is None:                  # 任一查不到就跳过
            continue
        _, rank1, pts1 = l1
        _, rank2, pts2 = l2

        # --- 特征 1: 积分占比(team1 积分占两人总和的比例, 越接近1越强) ---
        points_share = pts1 / (pts1 + pts2)
        # --- 特征 2: 双方近 5 场胜率(近期状态) ---
        f1 = win_rate_last_n(hist.get(p1, []), 5)
        f2 = win_rate_last_n(hist.get(p2, []), 5)
        # --- 特征 3: 历史交手胜率(贝叶斯平滑, 没交过手=0.5) ---
        k = tuple(sorted([p1, p2]))
        w1, w2 = h2h.get(k, [0, 0])
        h2h_w1 = w1 if p1 == k[0] else w2
        n = w1 + w2
        h2h_winrate = (h2h_w1 + 1) / (n + 2)          # (胜+1)/(总+2)
        # --- 特征 4: 排名差 / 积分差 / 赛事级别 / 主客场 ---
        rank_diff = rank1 - rank2                      # 负 => team1 排名更靠前
        points_diff = pts1 - pts2                      # 正 => team1 积分更多
        home_adv = int(to_bool(r["team1_at_home"])) - int(to_bool(r["team2_at_home"]))
        # home_adv: 1=team1是东道主, 0=中立, -1=team2是东道主

        rows.append({
            "date": d, "team1": r["team1"], "team2": r["team2"],
            "label": int(win1),                        # 标签: 1=team1胜, 0=team2胜
            "points_share": points_share,
            "h2h_winrate": h2h_winrate,
            "form5_1": f1, "form5_2": f2,
            "rank_diff": rank_diff, "points_diff": points_diff,
            "tier_level": tier_level(r["tier"]),
            "home_adv": home_adv,
        })

        # --- 用"当次结果"更新历史, 保证后续比赛只看到过去(防泄漏) ---
        hist.setdefault(p1, []).append(win1)
        hist.setdefault(p2, []).append(not win1)
        wins = h2h.get(k, [0, 0])
        wins[0 if (p1 == k[0]) == win1 else 1] += 1
        h2h[k] = wins

    return pd.DataFrame(rows)


if __name__ == "__main__":
    ms, rk = load_and_filter()
    print(f"过滤 2008-2022 后: 比赛 {len(ms)} 行, 排名(MS) {len(rk)} 行")

    df = build_features(ms, rk)
    print(f"生成特征表: {len(df)} 行 x {len(df.columns)} 列")
    print("\n前 3 行(看 features 和 label 长什么样):")
    print(df.head(3))

    os.makedirs("processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print("\n已保存到 processed/features.csv")
