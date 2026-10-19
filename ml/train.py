# -*- coding: utf-8 -*-
"""
ml/train.py —— 训练脚本(集成 MLflow 实验追踪)
对照 instructions 的要求, 每轮实验记录:
  - 参数: git commit SHA、数据集版本、模型类型、特征集
  - 指标: accuracy / AUC / F1
  - 制品: 混淆矩阵图 + 训练好的模型(pkl)

运行前先启动 MLflow UI(可选, 看实验面板):
    mlflow ui --port 5000
运行:
    python ml/train.py
"""
import subprocess
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")                    # 无界面环境下保存图
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from pathlib import Path

# 仓库根目录(本文件位于 ml/ 下)
ROOT = Path(__file__).resolve().parent.parent
# 让 stdout 以 UTF-8 输出(Windows 控制台默认 GBK, 打印中文/特殊字符可能报错)
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(encoding="utf-8", errors="replace")

SPLIT_YEAR = 2020

# v1 特征(基线): 积分占比/交手胜率/双方近5场
V1 = ["points_share", "h2h_winrate", "form5_1", "form5_2"]
# v2 特征(改进): v1 + 排名差/积分差/赛事级别/主客场
V2 = V1 + ["rank_diff", "points_diff", "tier_level", "home_adv"]


def get_git_sha():
    """取当前代码的 git commit SHA(instructions 要求记录的"代码版本")。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def run_experiment(run_name, model, features, dataset_version, train, test):
    """训练 + 评估 + 全部记录到 MLflow。"""
    with mlflow.start_run(run_name=run_name):
        # ---- 记录参数 ----
        mlflow.log_param("git_sha", get_git_sha())          # 代码版本
        mlflow.log_param("dataset_version", dataset_version)  # 数据版本(v1/v2)
        mlflow.log_param("model", type(model).__name__)       # 模型类型
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("split_year", SPLIT_YEAR)

        # ---- 训练 + 预测 ----
        model.fit(train[features], train["label"])
        proba = model.predict_proba(test[features])[:, 1]
        pred = (proba >= 0.5).astype(int)

        # ---- 记录指标 ----
        acc = accuracy_score(test["label"], pred)
        auc = roc_auc_score(test["label"], proba)
        f1 = f1_score(test["label"], pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)

        # ---- 记录制品: 混淆矩阵图 ----
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(test["label"], pred, ax=ax)
        cm_path = ROOT / "confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # ---- 记录制品: 模型文件 ----
        # 模型为我们自己训练, 显式信任 RF 的 Tree 类型(MLflow 3.x skops 安全扫描要求)
        mlflow.sklearn.log_model(
            model, "model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"[{run_name}] acc={acc:.3f}  auc={auc:.3f}  f1={f1:.3f}")
        return model


def main():
    # —— MLOps: 固定跟踪后端 ——
    # 用动态绝对路径直接连接 SQLite, 不经过 MLflow 相对路径解析
    # (MLflow 3.x 在 Windows 中文路径下会把路径 URL 编码, 导致建库失败)
    db_path = ROOT / "mlruns" / "mlflow.db"
    mlflow.set_tracking_uri("sqlite:///" + str(db_path).replace("\\", "/"))
    # 8 特征体系对应最终 app, 单独建实验, 与 13 特征研究实验区分
    mlflow.set_experiment("badminton-app-8feat")

    df = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["date"])
    train = df[df["date"].dt.year < SPLIT_YEAR]
    test = df[df["date"].dt.year >= SPLIT_YEAR]

    # 实验1: baseline 逻辑回归 + v1 特征
    run_experiment("baseline_lr_v1", LogisticRegression(max_iter=1000),
                   V1, "v1", train, test)

    # 实验2: improved 随机森林 + v2 特征
    run_experiment("improved_rf_v2",
                   RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0),
                   V2, "v2", train, test)

    # 保存"生产模型"(供 app/main.py 加载)
    final = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0)
    final.fit(train[V2], train["label"])
    model_path = ROOT / "model.pkl"
    joblib.dump({"features": V2, "model": final}, model_path)
    print(f"\n生产模型已保存到 {model_path}")


if __name__ == "__main__":
    main()
