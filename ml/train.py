# -*- coding: utf-8 -*-
"""
训练脚本：羽毛球男单比赛胜负预测（集成 MLflow）

实验设计：
- 数据：v2 特征数据（match_stats_v2.csv，由 ml/registry/data_pipeline.py
  从 v1 原始数据生成）。v1 原始数据只有比赛结果/文本字段、无特征，
  不能直接训练，故所有模型均在 v2 特征上训练。
- 切分：严格按时间顺序，最近 20% 比赛作为测试集（不随机打乱，避免未来信息泄露）。
- 实验：
  * baseline  = 逻辑回归（标准化，默认超参）
  * improved  = 随机森林（两组超参，各一个独立 run）
- 每次 run 自动记录：git commit SHA、数据版本、DVC 数据 md5、模型超参、
  accuracy/precision/recall/f1，以及模型、混淆矩阵图、分类报告等制品。

用法：
    python ml/train.py --all                         # 跑全部 3 个 run 并注册最佳
    python ml/train.py --data-version v2 --model logistic
    python ml/train.py --data-version v2 --model random_forest \
        --n-estimators 300 --max-depth 0 --register badmin-win-predictor
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def _to_path(uri):
    """把 file: artifact URI 转为本地 Path（兼容 file:/// 与 file: 盘符两种形式）。"""
    if uri.startswith("file://"):
        p = uri[len("file://"):]
    elif uri.startswith("file:"):
        p = uri[len("file:"):]
    else:
        p = uri
    p = unquote(p)
    if len(p) > 2 and p[0] == "/" and p[2] == ":":
        p = p[1:]
    return Path(p)

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
sys.path.insert(0, str(REPO_ROOT))
from ml.registry.data_pipeline import (  # noqa
    BASELINE_FEATURE_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, make_dataset,
)

EXPERIMENT_NAME = "badmin-match-prediction"
REGISTERED_MODEL = "badmin-win-predictor"
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# SQLite 数据库后端（MLflow 3.x 推荐，支持模型注册）；产物在 mlruns/，已在 .gitignore
TRACKING_DB = REPO_ROOT / "mlruns" / "mlflow.db"
TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")


def git_commit_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def dvc_md5(data_version):
    """从对应 .dvc 指针读取数据 md5（DVC 数据版本哈希）。"""
    path = DATA_DIR / f"match_stats_{data_version}.csv.dvc"
    if not path.exists():
        return "unknown"
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    return meta["outs"][0]["md5"]


def load_dataset(data_version):
    """v1 时现场跑数据管道（端到端可复现），v2 时读取固定 DVC 快照。"""
    if data_version == "v2":
        return pd.read_csv(DATA_DIR / "match_stats_v2.csv")
    if data_version == "v1":
        return make_dataset()  # 原始 v1 -> 特征（与 v2 快照同源）
    raise ValueError(data_version)


def time_split(df):
    """按日期排序后，最近 TEST_FRACTION 作为测试集。"""
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - TEST_FRACTION))
    return df.iloc[:cut], df.iloc[cut:]


def build_model(model_name, n_estimators, max_depth, min_samples_leaf):
    if model_name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       random_state=RANDOM_STATE)),
        ])
    if model_name == "random_forest":
        kwargs = {"n_estimators": n_estimators,
                  "min_samples_leaf": min_samples_leaf,
                  "random_state": RANDOM_STATE,
                  "n_jobs": -1}
        if max_depth and max_depth > 0:
            kwargs["max_depth"] = max_depth
        return RandomForestClassifier(**kwargs)
    raise ValueError(model_name)


def model_params(model_name, n_estimators, max_depth, min_samples_leaf):
    if model_name == "logistic":
        return {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"}
    return {"n_estimators": n_estimators, "max_depth": max_depth or 0,
            "min_samples_leaf": min_samples_leaf, "max_features": "sqrt",
            "random_state": 42}


def confusion_matrix_png(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Loss", "Pred Win"])
    ax.set_yticklabels(["True Loss", "True Win"])
    ax.set_title("Confusion Matrix (Team1)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_once(data_version, model_name, n_estimators, max_depth,
             min_samples_leaf, run_name=None, register=False, features=None):
    features = features or FEATURE_COLUMNS
    df = load_dataset(data_version)
    train, test = time_split(df)
    X_tr, y_tr = train[features], train[TARGET_COLUMN]
    X_te, y_te = test[features], test[TARGET_COLUMN]

    model = build_model(model_name, n_estimators, max_depth, min_samples_leaf)
    params = model_params(model_name, n_estimators, max_depth, min_samples_leaf)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        metrics = {
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
        }

        # 标签：当前代码版本
        mlflow.set_tag("git_commit_sha", git_commit_sha())
        mlflow.set_tag("project", "badminton-win-prediction")
        # 参数：数据版本 / DVC 哈希 / 模型 / 超参 / 切分
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("dvc_data_md5", dvc_md5(data_version))
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("features", ",".join(features))
        mlflow.log_params(params)
        mlflow.log_param("split", "time_series")
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("train_size", len(X_tr))
        mlflow.log_param("test_size", len(X_te))
        # 指标
        mlflow.log_metrics(metrics)
        # 制品
        report = classification_report(y_te, y_pred, zero_division=0,
                                       target_names=["Team1 Loss", "Team1 Win"])
        art_dir = _to_path(run.info.artifact_uri)
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        confusion_matrix_png(y_te, y_pred, art_dir / "confusion_matrix.png")
        with open(art_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(art_dir / "classification_report.txt")
        mlflow.log_artifact(art_dir / "confusion_matrix.png")
        mlflow.log_artifact(art_dir / "model.pkl")
        mlflow.sklearn.log_model(
            model, "model",
            input_example=X_te.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        registered = None
        if register:
            registered = register_model(run.info.run_id, REGISTERED_MODEL)

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model": model_name,
            "data_version": data_version,
            "dvc_md5": dvc_md5(data_version),
            "git_sha": git_commit_sha(),
            "params": params,
            "metrics": metrics,
            "registered_version": registered,
        }


def register_model(run_id, model_name):
    """注册模型到模型仓库，并给最佳版本打上 production 别名；返回版本号。"""
    try:
        result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        client = mlflow.MlflowClient()
        # MLflow 3.x 用别名（alias）替代旧的 Stage
        client.set_registered_model_alias(model_name, "production", result.version)
        return result.version
    except Exception as e:
        print("  [warn] 模型注册失败:", e)
        return None


def run_all():
    results = []
    # baseline：逻辑回归，仅用最朴素的 Elo 分差 + 主场（单一信号）
    results.append(run_once("v2", "logistic", 0, 0, 0,
                            run_name="baseline_logistic_elo",
                            features=BASELINE_FEATURE_COLUMNS))
    # improved：随机森林（全部工程特征），两组超参各一个独立 run
    results.append(run_once("v2", "random_forest", 300, 10, 20,
                            run_name="improved_rf_d10_l20",
                            features=FEATURE_COLUMNS))
    best = run_once("v2", "random_forest", 300, 12, 10,
                    run_name="improved_rf_d12_l10",
                    features=FEATURE_COLUMNS, register=True)
    results.append(best)

    print("\n=== 实验结果对比 ===")
    print(f"{'run':28s} {'model':14s} {'acc':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s}")
    for r in results:
        m = r["metrics"]
        print(f"{r['run_name']:28s} {r['model']:14s} "
              f"{m['accuracy']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['f1']:7.4f}")
    best_by_f1 = max(results, key=lambda r: r["metrics"]["f1"])
    print("\n最佳模型(F1):", best_by_f1["run_name"],
          "run_id=", best_by_f1["run_id"])
    summary = {"results": results, "best_run_id": best_by_f1["run_id"]}
    (REPO_ROOT / "ml" / "experiment_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果已写入 ml/experiment_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--model", default="logistic",
                    choices=["logistic", "random_forest"])
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=0)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="跑 baseline + 两个 RF 配置并注册最佳")
    args = ap.parse_args()

    if args.all:
        run_all()
    else:
        r = run_once(args.data_version, args.model, args.n_estimators,
                     args.max_depth, args.min_samples_leaf,
                     run_name=args.run_name, register=args.register)
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
