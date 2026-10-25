# -*- coding: utf-8 -*-
"""特征装配：把 M1 的编码结果整理成 M2/M3 直接可用的形状。

放在单独一层，是为了让「用什么特征」这件事可以随时换，而不动状态机本身。
三个函数对应三个下游模块：

    win_rate_vector()      给 M3（随机森林胜率）
    transition_key()       给 M2（马尔可夫链的转移统计键）
    search_context()       给 M4（MCTS 展开时的先验与掩码）
"""

from __future__ import annotations

from .heroes import HeroPool
from .state import DraftState

# 论文用的基础维度：我方 N 维 + 敌方 N 维
BASE_BLOCKS = ("ours", "opp")
# 建议增强的维度：禁用位、全局消耗、分路占用、系列赛进度
ENHANCED_BLOCKS = ("ours", "opp", "banned", "used", "lanes", "meta")
META_DIMS = 7


def win_rate_vector(state: DraftState, enhanced: bool = False) -> list[float]:
    """胜率模型的输入向量。

    enhanced=False  -> 2N 维，复刻论文（小样本时最不易过拟合）
    enhanced=True   -> 2N + ban + 全局消耗 + 分路 + 元特征，样本够多时再上
    """
    if not enhanced:
        return [float(v) for v in state.model_input(include_used=False)["x"]]
    out = state.model_input(
        include_banned=True, include_used=True, include_meta=True, include_lanes=True
    )
    return [float(v) for v in out["x"]]


def feature_spec(pool: HeroPool, enhanced: bool = False) -> dict:
    """告诉模型 / 前端每一段是什么，避免维度对不上时无从排查。"""
    n = pool.size
    blocks = [("ours", n), ("opp", n)]
    if enhanced:
        blocks += [
            ("banned", n),
            ("used_ours", n),   # 2026 规则：锁按队伍算，故两段分开
            ("used_opp", n),
            ("lanes", len(pool.lanes) * 2),
            ("meta", META_DIMS),
        ]
    return {
        "pool_size": n,
        "patch": pool.patch,
        "blocks": blocks,
        "dims": sum(d for _, d in blocks),
    }


def transition_key(state: DraftState) -> tuple:
    """轻量转移键：(步序, 行动方, 禁/选, 上一手英雄)。

    注意：M2 的正式实现（markov.TransitionModel）没有用这个键，而是用了
    「多阶回退 + 成对提升」的分解形式，原因见 markov.py 的模块说明。
    这个函数保留给两类场景：
      · 需要快速做一张「上一手 → 下一手」的可视化转移表（前端热力图）
      · 排位赛这类单一局数据里，做一个最朴素的 n-gram 基线
    """
    step = state.current_step
    last = state.last_action
    return (
        state.step_index,
        step.side if step else 0,
        step.kind if step else "done",
        last[0] if last else -1,
    )


def search_context(state: DraftState) -> dict:
    """M4 每展开一层需要的三样东西：合法动作、掩码、状态键。"""
    return {
        "state_key": state.state_key(),
        "legal": state.legal_hero_ids(),
        "available": state.available_mask(),
        "is_our_turn": state.is_our_turn,
        "kind": state.kind_to_act,
        "step_index": state.step_index,
    }
