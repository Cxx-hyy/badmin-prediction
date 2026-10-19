# -*- coding: utf-8 -*-
"""
app/main.py —— FastAPI 服务: 输入一场比赛的赛前特征, 输出 team1 胜率
=====================================================================
对照 instructions: 这是"ML 驱动的应用"的入口。
提供三个接口:
  GET  /health         健康检查(模型没加载时也要能返回, 不崩溃)
  POST /predict        输入 8 个特征 -> 返回 team1 胜率(0~1)和预测标签
  POST /predict_match  输入选手名+日期 -> 自动算特征 -> 返回 player1 胜率
"""
import os
from datetime import date as date_type

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.predict import build_match_features

app = FastAPI(title="Badminton Win Predictor")

# 模型路径用环境变量控制(CI/容器里可覆盖), 模型缺失时不崩溃
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

model = None
features = None


def load_model():
    """模型存在才加载; 不存在则保持 None, /health 会报告未就绪。"""
    global model, features
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        features = bundle["features"]


load_model()


class MatchFeatures(BaseModel):
    """一场比赛的赛前特征。加约束可做输入校验(对应测试的边界情况)。"""

    points_share: float = Field(ge=0, le=1, description="team1 积分占比")
    h2h_winrate: float = Field(ge=0, le=1, description="team1 历史交手胜率")
    form5_1: float = Field(ge=0, le=1, description="team1 近5场胜率")
    form5_2: float = Field(ge=0, le=1, description="team2 近5场胜率")
    rank_diff: float = Field(description="team1 排名 - team2 排名")
    points_diff: float = Field(description="team1 积分 - team2 积分")
    tier_level: int = Field(description="赛事级别分")
    home_adv: int = Field(ge=-1, le=1, description="主客场: 1/0/-1")


class MatchRequest(BaseModel):
    """按选手名预测的输入。tier 和 home_adv 可选。"""

    player1: str = Field(description="选手1 名字, 支持部分匹配, 如 Axelsen")
    player2: str = Field(description="选手2 名字")
    date: date_type = Field(description="比赛日期, YYYY-MM-DD")
    tier: str | None = Field(default=None, description="赛事级别字符串, 可选")
    home_adv: int | None = Field(default=None, ge=-1, le=1, description="主客场: 1/0/-1, 可选")


@app.get("/health")
def health():
    if model is None:
        return {"status": "model_not_loaded"}
    return {"status": "ok"}


@app.post("/predict")
def predict(m: MatchFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    x = pd.DataFrame([m.model_dump()])[features]  # 按训练时的特征顺序取列
    prob = model.predict_proba(x)[0][1]
    return {
        "team1_win_probability": round(float(prob), 4),
        "prediction": "team1" if prob >= 0.5 else "team2",
    }


@app.post("/predict_match")
def predict_match(req: MatchRequest):
    """输入两位选手名 + 日期, 自动构造 8 个特征并返回 player1 胜率。"""
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        res = build_match_features(req.player1, req.player2, req.date, req.tier, req.home_adv)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    f = res["features"]
    x = pd.DataFrame([f])[features]  # 按训练时的特征顺序取列
    prob = model.predict_proba(x)[0][1]
    return {
        "player1": res["player1"],
        "player2": res["player2"],
        "player1_win_probability": round(float(prob), 4),
        "prediction": "player1" if prob >= 0.5 else "player2",
        "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()},
    }
