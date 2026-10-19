# -*- coding: utf-8 -*-
"""端到端测试: 用临时训练的小模型, 验证预测方向正确。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient
import app.main as main

# 造一个"积分占比越高越可能赢"的玩具模型
X = np.array([[0.3], [0.5], [0.7], [0.9]] * 10)
y = (X[:, 0] > 0.5).astype(int)
m = LogisticRegression().fit(X, y)

# 注入到 app.main(覆盖真实模型)
main.model = m
main.features = ["points_share"]

client = TestClient(main.app)


def test_high_points_predicts_team1():
    # 积分占比高 -> 应判 team1
    r = client.post("/predict", json={
        "points_share": 0.9, "h2h_winrate": 0.5, "form5_1": 0.5, "form5_2": 0.5,
        "rank_diff": 0, "points_diff": 0, "tier_level": 5, "home_adv": 0})
    assert r.status_code == 200
    assert r.json()["prediction"] == "team1"


def test_low_points_predicts_team2():
    r = client.post("/predict", json={
        "points_share": 0.1, "h2h_winrate": 0.5, "form5_1": 0.5, "form5_2": 0.5,
        "rank_diff": 0, "points_diff": 0, "tier_level": 5, "home_adv": 0})
    assert r.json()["prediction"] == "team2"