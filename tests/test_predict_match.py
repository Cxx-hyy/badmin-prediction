# -*- coding: utf-8 -*-
"""测试按选手名预测: 人名解析 + 特征构造。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from ml.predict import FEATURES, build_match_features, resolve_player

# 排名表标准名的样例(用于纯函数测试, 不依赖大数据)
RK_NAMES = [
    "VIKTOR AXELSEN",
    "ANTHONY SINISUKA GINTING",
    "LEE CHONG WEI",
    "LEE ZII JIA",
    "CHONG WEI FENG",
]


# ---------- 人名解析 ----------
def test_resolve_player_partial():
    name, cand = resolve_player("Axelsen", RK_NAMES)
    assert name == "VIKTOR AXELSEN"
    assert cand == []


def test_resolve_player_full():
    name, cand = resolve_player("Anthony Sinisuka Ginting", RK_NAMES)
    assert name == "ANTHONY SINISUKA GINTING"
    assert cand == []


def test_resolve_player_ambiguous():
    name, cand = resolve_player("Lee", RK_NAMES)
    assert name is None
    assert len(cand) >= 2


def test_resolve_player_not_found():
    name, cand = resolve_player("Nobody", RK_NAMES)
    assert name is None
    assert cand == []


# ---------- 特征构造(依赖真实数据, 数据缺失则跳过) ----------
_DATA_MISSING = not (
    os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ms.csv"))
    and os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "bwf_historic_rankings.csv")
    )
)


@pytest.mark.skipif(_DATA_MISSING, reason="原始数据不存在, 跳过集成测试")
def test_build_match_features_axelsen_ginting():
    res = build_match_features("Axelsen", "Ginting", "2022-12-11")
    assert res["player1"] == "VIKTOR AXELSEN"
    assert res["player2"] == "ANTHONY SINISUKA GINTING"

    f = res["features"]
    assert set(f.keys()) == set(FEATURES)

    # 2022-12-11 之前最近一期排名(2022-12-05):
    #   Axelsen 第1名 116806 分; Ginting 第7名 70589 分
    assert f["rank_diff"] == 1 - 7
    assert f["points_diff"] == 116806 - 70589
    assert abs(f["points_share"] - 116806 / (116806 + 70589)) < 1e-9

    # 胜率类特征都在 0~1
    assert 0 <= f["form5_1"] <= 1
    assert 0 <= f["form5_2"] <= 1
    assert 0 <= f["h2h_winrate"] <= 1

    # 未提供可选输入 -> 用默认值
    assert f["tier_level"] == 5
    assert f["home_adv"] == 0
