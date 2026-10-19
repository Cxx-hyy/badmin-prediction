# -*- coding: utf-8 -*-
"""端到端校验 MLflow 实验记录、模型注册与 production 模型加载。

用法：
    python ml/verify_tracking.py
"""
import sys
from pathlib import Path

import mlflow
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.registry.data_pipeline import FEATURE_COLUMNS  # noqa: E402

MODEL_NAME = "badmin-win-predictor"
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name("badmin-match-prediction")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time"])
    print(f"=== 实验 {exp.name}：{len(runs)} 个 run ===")
    for r in runs:
        p, m, t = r.data.params, r.data.metrics, r.data.tags
        print(
            f"- {r.info.run_name}  model={p.get('model_type')} "
            f"features={p.get('n_features')} git={t.get('git_commit_sha','')[:10]} "
            f"dvc={p.get('dvc_data_md5','')[:10]}"
        )
        print(f"    acc={m.get('accuracy'):.4f} f1={m.get('f1'):.4f}")

    print("\n=== 注册模型 ===")
    rm = client.get_registered_model(MODEL_NAME)
    print(f"{MODEL_NAME} aliases={rm.aliases}")
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"  v{v.version} status={v.status} run={v.run_id[:10]}")

    print("\n=== 加载 production 模型预测 ===")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
    df = pd.read_csv(REPO_ROOT / "data" / "match_stats_v2.csv")
    sample = df[FEATURE_COLUMNS].tail(3)
    print("最近3场预测:", list(model.predict(sample)))
    print("真实标签:", list(df["label"].tail(3)))
    print("\n校验通过：实验记录、模型注册、production 加载均正常。")


if __name__ == "__main__":
    main()
