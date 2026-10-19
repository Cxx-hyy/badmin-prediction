# 模型与数据溯源（Provenance）

本文件记录从**原始数据**到**线上预测**的完整链路，确保任何一次预测都能回答：
"用的是哪份数据、哪版代码、哪次实验、哪个模型文件？"

> **最终应用链路（8 特征）**
>
> 最终交付的 FastAPI 应用采用 **8 特征模型**，其完整溯源链为：
> `data/raw/ms.csv` + `data/raw/bwf_historic_rankings.csv`（DVC md5
> `4dad64a7…` / `46edf8e4…`）→ `ml/data_pipeline.py` →
> `data/processed/features.csv`（md5 `f68e2d1a…`，9,641 行）→
> `ml/train.py`（MLflow 实验 `badminton-app-8feat`）→
> 仓库根 `model.pkl`（8 特征 bundle）→ `app/main.py` 加载。
> 详见 [`README_app_model.md`](README_app_model.md)。
>
> 下方"溯源链路总览"描述的是研究阶段的 **13 特征（Elo）体系**，
> 作为方法研究与模型注册实践保留。

## 溯源链路总览

```
原始比赛数据 ms.csv
        │  （纳入 DVC）
        ▼
data/match_stats_v1.csv ── md5 4dad64a7…，标签 data-v1
        │  （ml/registry/data_pipeline.py 特征工程）
        ▼
data/match_stats_v2.csv ── md5 b7c62be3…，标签 data-v2（13 特征）
        │  （ml/train.py 训练，MLflow 记录）
        ▼
MLflow run f99ecd4e…（improved_rf_d12_l10）
        │  （注册并打 production 别名）
        ▼
models:/badmin-win-predictor@production（v1）
        │  （python ml/export_model.py）
        ▼
models/model.pkl  ──►  FastAPI 推理服务（app/main.py，成员 A）
```

## 数据版本溯源

| 版本 | 本地文件 | DVC md5 | Git 标签 | 说明 |
|------|----------|---------|----------|------|
| v1 | `data/match_stats_v1.csv` | `4dad64a71f0ed5f03a89384ee38e66fa` | `data-v1` | 原始结果 28,270 场 |
| v2 | `data/match_stats_v2.csv` | `b7c62be31d2d78af250127a7086d18a7` | `data-v2` | 清洗+特征，21,136 场 |

- 真实文件存于 Azure Blob Storage（账户 `badminprediction`，容器 `dvc-data`，
  远程名 `azurestorage`），仓库里只有 `.dvc` 指针。
- 还原任意版本：`git checkout data-v1 && dvc checkout`（或 `data-v2`）。
- 校验当前数据与指针是否一致：`dvc status`；核对 md5：`dvc data status`。

## 实验 / 模型溯源（MLflow）

跟踪后端：`mlruns/mlflow.db`（SQLite）；实验名：`badmin-match-prediction`。

| run 名称 | run_id | 模型 | 特征 | F1 | 是否注册 |
|----------|--------|------|------|------|----------|
| baseline_logistic_elo | `b74deb85c4b94f5c89f8e7aeba707c21` | 逻辑回归 | 3 | 0.6617 | 否 |
| improved_rf_d10_l20 | `914d9a91b00e4a4c8fa9c555671d076e` | 随机森林 | 13 | 0.6616 | 否 |
| improved_rf_d12_l10 | `f99ecd4e7a6045d6aa3282188e41c9cc` | 随机森林 | 13 | 0.6646 | **v1 / production** |

每个 run 内都记录：

- `git_commit_sha`：`622d1368152b64b2649020160a27584cb1aeab4b`（训练代码提交）
- `dvc_data_md5`：`b7c62be31d2d78af250127a7086d18a7`（数据版本）
- `data_version`：`v2`；模型类型与全部超参；四项评估指标。
- 制品：`confusion_matrix.png`、`classification_report.txt`、`model.pkl`、模型 flavor 目录。

## 代码溯源

| 环节 | 文件 |
|------|------|
| 特征工程 | `ml/registry/data_pipeline.py`（`make_dataset()`，常量 `FEATURE_COLUMNS`） |
| 训练/记录/注册 | `ml/train.py` |
| 导出生产模型 | `ml/export_model.py` |
| 链路校验 | `ml/verify_tracking.py` |

## 如何验证整条链路

```bash
# 1) 确认数据与指针一致
dvc status
# 2) 校验 MLflow 记录、模型注册、production 加载是否正常
python ml/verify_tracking.py
# 3) 查看某次 run 的全部参数/指标/标签
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# 4) 完整复现（重生成 v2 → 训练 → 注册 → 导出）
python ml/registry/data_pipeline.py
python ml/train.py --all
python ml/export_model.py
```

## 重训与版本更新规则

1. 数据或代码变化后重新训练：新数据先 `dvc add` / `dvc push`，新代码先提交。
2. 每个 run 自动记录当时的 `git_commit_sha` 与 `dvc_data_md5`，保证可追溯。
3. 只有当新 run 的 **F1 不低于当前 production** 时才注册并切换别名；
   否则保留现版本，新 run 仅作记录。
4. 切换别名后重新 `python ml/export_model.py`，并更新 `ml/current_model.md`。
5. 数据文件、模型文件（`*.csv`、`models/`、`*.pkl`、`mlruns/`）均不进 Git，
   只由 DVC / MLflow 管理。
