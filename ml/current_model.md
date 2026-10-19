# 当前生产模型信息卡（current model）

> **重要说明 —— 最终应用实际加载的模型**
>
> 最终交付的 FastAPI 应用（`app/main.py`）加载的是仓库根 `model.pkl` 中的
> **8 特征随机森林 bundle**（基于 BWF 真实排名），完整说明见
> **[`README_app_model.md`](README_app_model.md)**。
>
> 本信息卡描述的是成员 C 在研究阶段完成、并通过 MLflow 注册的
> **13 特征模型（`badmin-win-predictor`，基于内部 Elo）**，作为方法研究与
> 模型注册实践保留；两套体系的对照见 `README_app_model.md` 第 7 节。

> 本文件描述当前线上（production）模型的全部关键信息。模型更新后请同步更新本卡。

## 身份信息

| 项 | 值 |
|----|----|
| 注册名称 | `badmin-win-predictor` |
| 版本 | v1 |
| 别名 | `production` |
| 模型类型 | 随机森林 `RandomForestClassifier`（sklearn） |
| MLflow run_id | `f99ecd4e7a6045d6aa3282188e41c9cc` |
| 实验 | `badmin-match-prediction` |
| run 名称 | `improved_rf_d12_l10` |

## 超参数

| 参数 | 值 |
|------|----|
| `n_estimators` | 300 |
| `max_depth` | 12 |
| `min_samples_leaf` | 10 |
| `max_features` | `sqrt` |
| `random_state` | 42 |

## 数据与代码版本

| 项 | 值 |
|----|----|
| 数据版本 | v2（`data/match_stats_v2.csv`，21,136 行） |
| DVC 数据 md5 | `b7c62be31d2d78af250127a7086d18a7` |
| 数据标签 | `data-v2` |
| 训练代码 commit | `622d1368152b64b2649020160a27584cb1aeab4b` |
| 训练/测试切分 | 按时间前 80% / 最近 20%（16,908 / 4,228 场） |

## 测试集指标（最近 20%，2023.09–2026.07）

| 指标 | 值 |
|------|----|
| accuracy | 0.6717 |
| precision | 0.6704 |
| recall | 0.6588 |
| **F1** | **0.6646** |

## 输入特征（顺序必须一致，共 13 个）

```
elo_diff, win_rate_diff, recent_form_1, recent_form_2,
h2h_matches, h2h_win_rate_1,
point_margin_1, point_margin_2, point_margin_diff,
game_win_rate_1, game_win_rate_2,
at_home_1, at_home_2
```

输出：`1` 表示 team1 获胜，`0` 表示 team1 落败（也可调用 `predict_proba` 取胜率）。

## 模型制品

- MLflow 注册中心：`models:/badmin-win-predictor@production`
- 导出的离线文件（**不进 Git**，`models/` 已被忽略）：
  - `models/model.pkl`：sklearn 模型（已含训练流水线）
  - `models/model_metadata.json`：本卡对应的机读元数据

## 加载与推理

**方式一：从 MLflow 注册中心加载（推荐，自动拿到 production 版本）**

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
model = mlflow.pyfunc.load_model("models:/badmin-win-predictor@production")
model.predict(feature_df)      # 0/1
model.predict_proba(feature_df) # 概率
```

**方式二：直接加载导出的 pickle**

```python
import pickle
model = pickle.load(open("models/model.pkl", "rb"))
model.predict(feature_df)
```

## 给推理服务（成员 A）的约定

- 线上特征**必须**由 `ml/registry/data_pipeline.py` 的 `make_dataset()` 计算，
  训练与推理使用同一份特征逻辑，保证特征含义一致；接口构造好对阵双方后，
  从该模块取同一行特征即可。
- 特征列顺序以 `FEATURE_COLUMNS` 为准（与上面列表一致），不要自行调整顺序。
- `label=1` 表示 team1 胜；接口返回时请把 team1/team2 与预测结果、概率一并返回。
- 数据缺失时：无历史记录的胜率/交手按 0.5 处理；主客场缺失按非主场（0）处理。
- 模型重训并出现更优版本后，由成员 C 注册新版本、把 `production` 别名切到新版本，
  并重新运行 `python ml/export_model.py` 刷新 `models/`，服务端无需改代码即可加载。
