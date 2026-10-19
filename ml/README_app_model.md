# 最终应用模型说明（8 特征 · FastAPI 服务）

本文件说明**最终交付的 FastAPI 应用（`app/main.py`）实际加载的模型**：
一套基于 **BWF 真实排名 + 近期状态** 的 **8 特征**胜负预测模型。
本系统由成员 A/B 设计应用与接口，成员 C 负责把数据、特征、训练、追踪、模型交付
这条 MLOps 链路补完并验证，使整个应用端到端可运行、可复现。

> 项目中另有一套成员 C 早期完成的 **13 特征研究体系**（基于内部 Elo 评分，
> 记录在 `README_experiments.md` / `current_model.md`）。两套体系的关系见文末。

---

## 1. 八个特征（顺序固定，即应用接口）

| # | 特征 | 含义 |
|---|------|------|
| 1 | `points_share` | team1 积分占两队总积分的比例（0~1） |
| 2 | `h2h_winrate` | team1 历史交手胜率（0~1，无交手记录取 0.5） |
| 3 | `form5_1` | team1 近 5 场胜率（0~1） |
| 4 | `form5_2` | team2 近 5 场胜率（0~1） |
| 5 | `rank_diff` | team1 排名 − team2 排名（负数表示 team1 排名更高） |
| 6 | `points_diff` | team1 积分 − team2 积分 |
| 7 | `tier_level` | 赛事级别分（Super 1000=7 … 未知兜底=2） |
| 8 | `home_adv` | 主客场：team1 主场=1、中立=0、客场=−1 |

所有特征均只使用**比赛开始前**的历史信息，不存在未来信息泄露。

---

## 2. 数据链路（DVC 版本化）

原始数据不进 Git，由 DVC + Azure Blob 管理，仓库只存 `.dvc` 指针：

| 数据 | 路径 | 规模 | DVC md5 |
|------|------|------|---------|
| 比赛结果 | `data/raw/ms.csv` | 28,270 场（本管线过滤后 21,646 行） | `4dad64a71f0ed5f03a89384ee38e66fa` |
| BWF 历史排名 | `data/raw/bwf_historic_rankings.csv` | 61.7 MB（过滤后 MS 73,700 行） | `46edf8e4f8f4719ca3f20ed3c6fd3472` |
| 8 特征训练表 | `data/processed/features.csv` | 9,641 行 × 12 列 | `f68e2d1a23c0843ead78e88bd408be53` |

- 时间范围：**2008-01-01 ~ 2022-12-31**（`ml/data_pipeline.py` 的 START/END）。
- 特征表标签分布：team1 胜 5,029 / 负 4,612，类别均衡。
- 特征表列：`date, team1, team2, label` + 8 个特征。

---

## 3. 训练与评估（baseline vs improved）

- **划分**：按时间顺序，**不随机打散**。`SPLIT_YEAR=2020`：
  - 训练集 8,563 场（2008–2019），测试集 1,078 场（2020–2022）。
- **baseline（v1，4 特征）**：逻辑回归，特征为
  `points_share, h2h_winrate, form5_1, form5_2`。
- **improved（v2，8 特征）**：随机森林
  `n_estimators=300, max_depth=6, random_state=0`，新增
  `rank_diff, points_diff, tier_level, home_adv`。

### 实验结果（MLflow 实验 `badminton-app-8feat`）

| run | 模型 | 特征数 | accuracy | AUC | F1 |
|-----|------|------|----------|------|------|
| `baseline_lr_v1` | LogisticRegression | 4 | 0.7226 | 0.7937 | **0.7379** |
| `improved_rf_v2` | RandomForest 300/深度6 | 8 | **0.7236** | 0.7898 | 0.7363 |

### 结果的客观分析

- 两模型准确率都在 **72% 左右**，F1 约 **0.74**，说明"积分占比 + 近期状态"
  已能解释胜负的大部分信息。
- 增加 4 个特征并换成随机森林后，**accuracy 仅微升（+0.001），F1/AUC 反而略降**，
  没有统计意义上的显著提升。这是真实数据的诚实结论：
  - 排名/积分信息与 `points_share` 高度共线，边际增益小；
  - 赛事级别、主客场对胜负影响很弱（羽毛球多为中立场地）。
- 最终应用仍采用 **8 特征随机森林**作为交付模型：它纳入了语义更完整的特征，
  且在不牺牲准确率的前提下更稳健，`/predict_match` 按选手名自动取数时特征更一致。

---

## 4. 模型交付（bundle）

训练脚本把生产模型打包成一个 **bundle 字典**，写入仓库根 `model.pkl`：

```python
{"features": [<8 个特征名>], "model": <RandomForestClassifier>}
```

- `app/main.py` 在启动时 `joblib.load("model.pkl")`，分别取
  `bundle["model"]` 与 `bundle["features"]`，并按训练时的特征顺序取列。
- 模型路径可用环境变量 `MODEL_PATH` 覆盖；模型缺失时 `/health` 返回
  `model_not_loaded`，服务不崩溃。

---

## 5. 端到端验证结果

- **pytest：16 passed**（数据就位后，原先 skip 的 Axelsen–Ginting 特征构造测试转为通过）。
- **干净进程真实模型验证**（不经过测试夹具注入的玩具模型）：
  - `GET /health` → `{"status": "ok"}`；
  - `POST /predict`（8 特征）→ 胜率 0.694，预测 `team1`；
  - `POST /predict_match`（Axelsen vs Ginting，2022-12-11）→
    胜率 0.831，预测 `player1`，特征正确：
    `points_share=0.6233, rank_diff=-6, points_diff=46217, form5_1=0.8, form5_2=1.0`。

> 注意：`tests/test_integration.py` 会注入一个临时训练的玩具逻辑回归到
> `app.main`，用于方向性测试；这会在同一测试会话里影响后续 API 测试。
> 因此验证"真实模型"必须在**独立进程**中进行，不能只看测试会话的结果。

---

## 6. 如何复现

```bash
# 1. 拉取数据（需 Azure 连接串，找成员 C，勿提交到仓库）
$env:AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=...;AccountName=...;AccountKey=..."
dvc pull

# 2. 从原始数据生成 8 特征表（在仓库根运行）
python ml/data_pipeline.py

# 3. 训练 baseline + improved，记录 MLflow，并导出 model.pkl
python ml/train.py

# 4. 运行测试
pytest

# 5. 启动服务
uvicorn app.main:app --reload
```

---

## 7. 与 13 特征研究体系的关系

| 维度 | 8 特征（最终应用） | 13 特征（研究体系） |
|------|-------------------|---------------------|
| 实力刻画 | BWF **真实排名/积分** | 数据管道内部计算的 **margin-adjusted Elo** |
| 数据范围 | 2008–2022（9,641 场） | 2007–2026（21,136 场） |
| 测试集 | 2020–2022（1,078 场） | 最近 20%（2023–2026，4,228 场） |
| 模型 | RF 300/深度6，bundle 交付 | RF 300/深度12，注册 `badmin-win-predictor` |
| 追踪 | 实验 `badminton-app-8feat` | 实验 `badmin-match-prediction` |
| 用途 | **FastAPI 应用实际加载** | 方法研究与 MLflow 模型注册实践 |

两套体系的测试集与时间范围不同，**指标不可直接横向比较**；它们共同体现了
"特征工程 + 实验追踪 + 模型版本化"的完整 MLOps 方法，最终应用以 8 特征为准。
