# -*- coding: utf-8 -*-
"""把 MLflow 中 production 别名的生产模型导出到 models/，供推理服务加载。

用法：
    python ml/export_model.py

产出（models/ 已被 .gitignore 忽略，不进 Git）：
    models/model.pkl          sklearn 生产模型（含预处理流水线）
    models/model_metadata.json 模型信息（数据版本、特征顺序、指标、run_id 等）
"""
import json
import pickle
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.registry.data_pipeline import FEATURE_COLUMNS  # noqa: E402

MODEL_NAME = "badmin-win-predictor"
MODEL_DIR = REPO_ROOT / "models"
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    # 取 production 别名对应的版本
    mv = client.get_model_version_by_alias(MODEL_NAME, "production")
    run_id = mv.run_id
    run = client.get_run(run_id)
    p, m = run.data.params, run.data.metrics

    # 用 sklearn flavor 加载并导出
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "registered_name": MODEL_NAME,
        "version": mv.version,
        "alias": "production",
        "model_type": p.get("model_type"),
        "run_id": run_id,
        "data_version": p.get("data_version"),
        "dvc_data_md5": p.get("dvc_data_md5"),
        "git_commit_sha": run.data.tags.get("git_commit_sha"),
        "hyperparams": {
            "n_estimators": int(p.get("n_estimators", 0) or 0),
            "max_depth": int(p.get("max_depth", 0) or 0),
            "min_samples_leaf": int(p.get("min_samples_leaf", 0) or 0),
            "max_features": p.get("max_features"),
        },
        "features": FEATURE_COLUMNS,
        "metrics": {
            "accuracy": round(m.get("accuracy", 0), 4),
            "precision": round(m.get("precision", 0), 4),
            "recall": round(m.get("recall", 0), 4),
            "f1": round(m.get("f1", 0), 4),
        },
        "load_example": "import pickle; model = pickle.load(open('models/model.pkl','rb'))",
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"生产模型已导出: {MODEL_DIR / 'model.pkl'}")
    print(f"  run_id={run_id}  version=v{mv.version}")
    print(f"  指标: {metadata['metrics']}")
    print(f"  特征数: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
