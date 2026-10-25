---
name: gc7-bpcoach
description: 王者荣耀 KPL 2026 全局 B/P（禁选）推荐引擎。当用户询问 KPL 比赛的 BP、禁选、下一手该 ban/pick 谁、阵容强度对比，或需要按 2026 全局 B/P 规则（20 手、每方每英雄限用一次）分析对局时使用。This skill should be used for Honor of Kings KPL draft (ban/pick) recommendations and lineup strength analysis under 2026 global-BP rules. 输出定位为「第二意见」，不是胜率承诺。
agent_created: true
---

# Gc7bpcoach：KPL 2026 B/P 推荐

## 能力

基于 1447 局官方比赛训练的四层模型（M1 状态机 / M2 马尔可夫选人预测 / M3 阵容胜率 / M4 树搜索），给定任意 B/P 局面输出推荐下一手、备选排序、对手应对与参照胜率。英雄池 134 张（S45），规则为 KPL 2026 全局 B/P（20 手，每方每英雄限用一次，元流之子 5 张职业卡共用一个选用名额）。

## 运行前提

1. 准备 Python 3.11+，安装依赖：`pip install -r requirements.txt`（钉在 `scikit-learn>=1.9,<2`：模型是 joblib pickle，跨大版本加载可能失效）。
2. 模型文件已随仓库附带：`models/bpcoach_v0.5.0.bpc`（132 KB，含 sha256 清单，加载时自动校验，被改动会拒绝加载）。

## 用法

运行 `scripts/recommend.py`，传入一个描述局面的 JSON 字符串：

```bash
python scripts/recommend.py '{"our_side":"blue"}'
```

局面 JSON 字段：

- `our_side`（必填）：`"blue"` 或 `"red"`，我方持边。
- `steps`（可选）：本局已走的手，按顺序给英雄名（支持常见别名），如 `["盾山","大乔","赵云"]`。ban/pick 由模板自动判定，不要标注。
- `prior_games`（可选）：前几局记录 `[{"picks_blue":[...], "picks_red":[...], "winner":"blue"|"red"}]`。打第 2 局及以后必须提供，否则全局锁（已用过的英雄不能再选）不生效。
- `depth`（可选）：搜索深度，默认 3；赶时间用 2。
- `opp_risk`（可选）：对手口径 0.0~1.0，默认 0（按 M2 概率加权）；1 表示假设对手总选最坏应对。
- `m3`（可选）：`"lineup"`（默认，只看阵容）/ `"team"` / `"state"`。

局中示例（第 2 局、当前第 5 手、轮到我方选）：

```bash
python scripts/recommend.py '{"our_side":"red","steps":["盾山","大乔","赵云","关羽"],"prior_games":[{"picks_blue":["鲁班大师","姜子牙","卢雅那","梦奇","刘备"],"picks_red":["墨子","司空震","孙权","王维","心魔六耳"],"winner":"blue"}]}'
```

## 输出解读

- `recommendation` 与 `candidates`：推荐与备选。`rank_by` 标识排序依据——选人步为 `"value"`（搜索值，晋级概率口径），禁人步为 `"denial"`（拒绝价值：禁掉这张卡能把对手上限压低多少，单位是百分点的小数，如 0.0081 = 0.81pt）。
- `is_our_turn` 为 `false` 时，输出的是「对方这手最可能/最凶的应法」，用于预判而非照抄。
- `degenerate: true` 表示所有候选同值，「并列第一」不构成推荐，换深度或口径再看。
- `ok: false` 时读 `error` 字段修正输入（常见：英雄名打错、把已用过的英雄又选了一遍）。

## 能力边界（照实告知用户）

- 阵容评估模型的交叉验证准确率约 54.7%（LOSO），候选之间的差距常小于噪声底。输出是**可复现、可辩论的第二意见**：附拒绝价值、对手应对与跨局代价，供教练组讨论，不承诺提高胜率。
- 不含英雄间克制/交互项，不含第 7 局巅峰对决（盲选无 ban）。
- 数据来自公开赛事战报，仅供研究使用。
