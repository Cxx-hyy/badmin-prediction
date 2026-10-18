# 数据集说明（羽毛球男单比赛胜负预测）

本项目使用 DVC + Azure Blob Storage 管理数据集。数据文件本身（`.csv`）**不进 Git**，
仓库中只保存体积很小的 `.dvc` 指针文件；真实数据由 `dvc pull` / `dvc push`
从 Azure 远程存储 `azurestorage`（容器 `dvc-data`）拉取或上传。

## 数据来源

- 真实羽毛球男单（MS）比赛结果数据，时间跨度 **2007–2026**，共 28,270 场。
- 原始文件由成员 A 提供，纳入 DVC 后命名为 `match_stats_v1.csv`。
- 另有一份 BWF 历史排名数据（约 61MB）经评估**不纳入**项目：其选手姓名与
  比赛数据的直接匹配率仅约 15.5%，而所需的排名信息已由数据管道内部计算的
  Elo 评分等价替代，避免了高成本的姓名模糊匹配。

## 数据版本总览

| 版本 | 文件名 | 行数 | 说明 | Git 标签 |
|------|--------|------|------|----------|
| v1 | `data/match_stats_v1.csv` | 28,270 | 原始比赛结果，含文本字段与缺失值 | `data-v1` |
| v2 | `data/match_stats_v2.csv` | 21,136 | 清洗 + 特征工程后的训练数据（13 个建模特征） | `data-v2` |

> 标签即版本锚点，提交哈希可用 `git rev-parse data-v1` / `git rev-parse data-v2`
> 查询（以 GitHub 远端标签为准）；`git checkout data-v2` 后 `dvc checkout` 即可还原对应数据。

## v1 原始字段（`match_stats_v1.csv`）

| 字段 | 含义 |
|------|------|
| `date` | 比赛日期 |
| `discipline` | 项目（均为 MS 男单） |
| `tournament` / `tier` | 赛事名称 / 赛事层级（Super 100…1000、Grand Prix 等） |
| `round` | 比赛轮次（缺失 65 行） |
| `host_location` | 举办地（缺失 1,435 行） |
| `team1` / `team2` | 双方选手姓名 |
| `winner` | 胜者（1 = team1，2 = team2） |
| `score` | 每局比分（如 `21-15 17-21 21-16`） |
| `team1_at_home` / `team2_at_home` | 双方是否主场（布尔） |

## v2 特征（`match_stats_v2.csv`）

由 `ml/registry/data_pipeline.py` 从 v1 生成，**所有特征均只使用比赛开始前的历史信息**，
按比赛时间顺序单遍计算，杜绝未来信息泄露。

**建模特征（`FEATURE_COLUMNS`，13 个）**：

| 特征 | 含义 |
|------|------|
| `elo_diff` | 双方赛前 **margin-adjusted Elo** 分差（初始 1500，K=32，大胜多涨分） |
| `win_rate_diff` | 双方历史总胜率之差 |
| `recent_form_1` / `recent_form_2` | 双方近 10 场胜率（近期状态） |
| `h2h_matches` / `h2h_win_rate_1` | 双方历史交手次数 / team1 交手胜率 |
| `point_margin_1` / `point_margin_2` / `point_margin_diff` | 双方历史场均局点差（每局净胜分）及差值 |
| `game_win_rate_1` / `game_win_rate_2` | 双方历史局胜率（赢局/总局数） |
| `at_home_1` / `at_home_2` | 双方是否主场（0/1） |

**baseline 朴素特征（`BASELINE_FEATURE_COLUMNS`，3 个）**：`elo_diff`、`at_home_1`、`at_home_2`。

**标签**：`label` = team1 获胜为 1，否则为 0。

此外 v2 保留了 `elo_1/elo_2`、`win_rate_1/win_rate_2`、`matches_1/matches_2`、
`date/team1/team2` 等信息列，便于解释与接口展示，默认不进入训练。

清洗与样本筛选规则：
- 解析日期、按时间排序，删除关键字段缺失或 `winner` 非法的行；主客场缺失填 False。
- 仅保留双方在赛前各至少有 5 场历史的比赛（保证特征有意义），故 28,270 → 21,136 行。
- 无历史信息的胜率/交手记录以 0.5（中立值）填充。
- margin-adjusted Elo 依据比分 `score` 解析出的局点差，对大胜方放大 K，
  以区分"21-10 大胜"与"21-19 险胜"。

## 数据质量与基线（按时间切分，最近 20% 为测试集）

- v2 无缺失值；标签分布均衡（胜 10,731 / 负 10,405，约 50.8% / 49.2%）。
- 最近三个赛季（2023-09 起，4,228 场）上：
  - 仅用 Elo 阈值的朴素准确率约 66.5%（该任务主要受 Elo 主导，可预测性上限由此决定）。
  - baseline 逻辑回归（仅 Elo+主场）：accuracy ≈ 0.6649，F1 ≈ 0.6617。
  - improved 随机森林（全部 13 特征 + 调参）：accuracy ≈ 0.6717，F1 ≈ 0.6646，
    在准确率与 F1 上均优于 baseline。
- 主客场特征相关性极弱，符合羽毛球以中立场地为主、个人主场优势不明显的实际。

## DVC 常用操作

```bash
# 首次拉取 / 同步数据（需要 Azure 连接串，找成员 C 索取，勿提交到仓库）
# Windows PowerShell:
$env:AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
dvc pull

# 新增/更新数据
dvc add data/match_stats_xx.csv
dvc push
git add data/match_stats_xx.csv.dvc
git commit -m "feat(data): ..."

# 切换到某个数据版本后同步数据文件
git checkout data-v2
dvc checkout
```

## 重新生成 v2

```bash
python ml/registry/data_pipeline.py
```

读取 `data/match_stats_v1.csv`，输出 `data/match_stats_v2.csv`。
