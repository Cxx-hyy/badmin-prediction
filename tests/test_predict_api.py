# -*- coding: utf-8 -*-
"""测试 FastAPI 接口: 合法输入、缺字段、非法值、A/B 互换对称性。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

#测试时可更改VALID, 分别是积分占比、历史交手胜率、近5场胜率、排名差、积分差、赛事级别、主客场
VALID = {
    "points_share": 0.586, "h2h_winrate": 0.6,
    "form5_1": 0.8, "form5_2": 0.4,
    "rank_diff": -5, "points_diff": 12000,
    "tier_level": 7, "home_adv": 0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "model_not_loaded")


def test_valid_predict():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    p = r.json()["team1_win_probability"]
    assert 0 <= p <= 1                      # 概率必须在 0~1


def test_missing_field():
    bad = {k: v for k, v in VALID.items() if k != "points_share"}  # 删一个字段
    r = client.post("/predict", json=bad)
    assert r.status_code == 422             # 缺字段 -> 422 而不是 500


def test_invalid_value():
    bad = dict(VALID, form5_1=2.0)          # 胜率不可能 > 1
    r = client.post("/predict", json=bad)
    assert r.status_code == 422             # 非法值 -> 422


def test_symmetry():
    # A/B 互换, 概率应互补(≈1)
    swap = dict(VALID,
                points_share=1 - VALID["points_share"],
                h2h_winrate=1 - VALID["h2h_winrate"],
                form5_1=VALID["form5_2"], form5_2=VALID["form5_1"],
                rank_diff=-VALID["rank_diff"],
                points_diff=-VALID["points_diff"],
                home_adv=-VALID["home_adv"])
    p1 = client.post("/predict", json=VALID).json()["team1_win_probability"]
    p2 = client.post("/predict", json=swap).json()["team1_win_probability"]
    assert abs(p1 + p2 - 1) < 0.2           # 允许小误差(树模型不对称)