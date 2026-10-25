# Gc7bpcoach

王者荣耀 KPL 2026 全局 B/P（禁选）推荐引擎的 WorkBuddy Skill 封装。
给定任意 B/P 局面，输出推荐下一手、备选排序、对手应对与参照胜率。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/recommend.py '{"our_side":"blue"}'
```

输出示例（开局第一手禁人）：

```json
{
  "ok": true,
  "rank_by": "denial",
  "recommendation": {"hero": "赵云", "denial": 0.0081, "lane": "打野"},
  ...
}
```

更多局面参数（局中、跨局全局锁、搜索深度、对手口径）见 [SKILL.md](SKILL.md)。

## 模型

- 四层结构：M1 局面状态机 → M2 马尔可夫选人预测（留一 top1 16.8%）→
  M3 阵容胜率（LOSO 54.7%）→ M4 树搜索推荐。
- 训练数据：KPL 2026 官方比赛 1447 局；英雄池 134 张（S45）。
- 模型本体为单文件推理包 `models/bpcoach_v0.5.0.bpc`（zip，含 sha256 清单，
  加载时校验）。

## 定位与免责

本模型的输出是**可复现、可辩论的「第二意见」**：阵容评估分辨率约 54.7%，
候选间差距常小于噪声底，不构成胜率承诺。数据来自公开赛事战报，仅供研究使用。

## License

代码与模型仅供学习研究，转载请注明出处。
