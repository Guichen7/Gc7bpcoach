# -*- coding: utf-8 -*-
"""M2 选人预测：在 M1 的局面之上，给出「对手下一手会选/禁谁」的概率分布。

一、模型是什么
论文里 M2 是一条马尔可夫链：状态 = 当前 B/P 棋盘，动作 = 下一个英雄，
用历史对局的转移次数估 P(动作 | 状态)。直接照抄会遇到一个硬问题：
117 维棋盘的状态空间是组合爆炸，一局一个状态的样本量下几乎全部为 0。

所以这里用**分层 + 分解**两件事把链做实用：

  1. 分层（backoff）：状态从细到粗有多个投影，
        (步序, 动作方, 禁/选, 上一手)  →  (步序, 动作方, 禁/选)  →  (动作方, 禁/选)  →  (禁/选)
     细层样本不够就退到粗层。这是稀疏马尔可夫链的标准处理，也叫可变阶马尔可夫。

  2. 分解（log-linear）：把「状态」拆成它包含的英雄集合，用二阶统计量代替高阶联合：
        score(x) = w_m · log p_marg(x | 禁/选, 方)          ← 一阶边际
                 + w_s · mean_{y∈我方已选} logLift(x, y)     ← 同队协同
                 + w_c · mean_{y∈对方已选} logLift(x, y)     ← 跨队克制
                 + w_b · mean_{y∈已 ban}   logLift(x, y)     ← 被 ban 后的替代
                 + w_l · lane_fit(x)                        ← 分路缺口
     再对合法动作集做 softmax。这等于把马尔可夫链的转移核参数化成
     「边际 × 成对提升」的乘积形式（Hammersley-Clifford 式的伪似然估计），
     好处是**不需要真实手顺序**也能从「一局用了哪 20 个英雄」里学到东西。

     这个分解最初是为了「只有集合、没有手顺序」的战报数据才做的。
     现在数据换成了官方接口的真实手顺序（dataset.order_known=true），第 1 条的分层
     真正吃到了样本，w_position 也打开了 —— 但分解本身仍然留着：
     它把 134 维棋盘的高阶联合换成了可估的二阶统计量，在 1447 局这个量级上
     比直接估转移矩阵稳得多（见 tools/train_markov.py 的消融）。

二、为什么需要 lane_fit
职业 B/P 有一条硬常识：五个位置各一人。到第 5 手时，还没补的位置基本决定了选谁。
把它做成软约束（加分项）而不是硬掩码，是为了保留「非常规分路」的可能，
同时在样本稀疏时提供一个可靠的兜底信号。

    这里有个必须小心的细节：**摇摆位**（马超可对抗可打野）不能按「两条路都算占满」处理，
    否则一个英雄会同时填掉两个位置，模型会误判阵容已齐。所以 lane_fit 的输入不是
    「这英雄能打哪几条路」的 0/1 集合，而是 p(lane | hero) 的概率分布，再加一次
    单射分配（一个英雄只能占一条路，见 lane.py）。摇摆英雄的加分因此自动被摊薄：
    有打野时马超仍能补对抗（0.75），纯打野则几乎为 0（0.05）。

三、输出
    predict(state, top_k=3) -> Prediction
        .ranked   [(hero_id, prob)] 降序，只含合法动作
        .top_k    论文里进 MCTS 的前 k 个候选
        .entropy  分布越平说明模型越没把握，M4 可以据此调整搜索宽度
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Iterable

from .dataset import Dataset, GameSample, SeriesSample
from .heroes import HeroPool
from .series import BLUE, RED, other
from .state import DraftState
from .template import BAN, PICK, BPTemplate

# 成对统计的关系类型
REL_PICK_SAME = "pick_same"     # 同队两个选人（协同）
REL_PICK_CROSS = "pick_cross"   # 跨队两个选人（克制/应对）
REL_BAN = "ban"                 # 两个 ban 同时出现（同一套禁用思路）
REL_BAN_PICK = "ban_pick"       # ban 与选人共现（被针对的强势英雄）


@dataclass
class MarkovConfig:
    """权重与平滑系数。

    默认值来自 tools/train_markov.py 的留一系列交叉验证扫描
    （当前口径：347 折、28940 个决策点，1447 局）：
        分路缺口（w_lane）是最稳的信号，top3 从 28.3% 提到 34.7%；
        成对项（协同/克制）在 1447 局上从负收益转为正收益，top1 13.7% → 15.5%；
        w_lane 的量纲说明：lane_fill 的取值从旧的 0/1/2 改成 0~1 之后，
        权重同步放大到 3.0 才是等价强度，不要按旧值 1.0 去理解。
    """

    w_marginal: float = 1.0
    w_synergy: float = 0.30      # 我方已选 → 下一手协同
    w_counter: float = 0.25      # 对方已选 → 下一手反制
    w_ban: float = 0.25          # 已 ban 集合的联合模式
    w_ban_pick: float = 0.35     # ban 之后的替代选择
    w_lane: float = 3.0          # 分路缺口（当前数据下最稳的信号；量纲从 0/1/2 改成 0~1 后期权重同步放大）
    w_position: float = 0.8      # 位置先验（仅 order_known 数据有效）
    alpha_marginal: float = 0.4  # 边际平滑（向均匀分布收缩）
    alpha_pair: float = 0.3      # 成对提升平滑
    clip_log_lift: float = 2.0   # log-lift 截断，防止小样本爆掉
    temperature: float = 1.0


@dataclass
class Prediction:
    ranked: list[tuple[int, float]]
    top_k: list[tuple[int, float]]
    entropy: float
    n_legal: int
    step_index: int
    kind: str | None
    side: int | None

    def prob_of(self, hero_id: int) -> float:
        for h, p in self.ranked:
            if h == hero_id:
                return p
        return 0.0

    def rank_of(self, hero_id: int) -> int | None:
        for i, (h, _) in enumerate(self.ranked, start=1):
            if h == hero_id:
                return i
        return None

    def names(self, pool: HeroPool, k: int = 3) -> list[str]:
        return [f"{pool.name_of(h)} {p:.1%}" for h, p in self.top_k[:k]]


class TransitionModel:
    """M2 的主体。fit 之后不可变，可安全保存/加载。"""

    def __init__(
        self,
        pool: HeroPool,
        config: MarkovConfig | None = None,
        template_id: str | None = None,
    ) -> None:
        self.pool = pool
        self.config = config or MarkovConfig()
        self.template_id = template_id

        n = pool.size
        self.marg: dict[tuple[str, int], list[float]] = {
            (kind, side): [0.0] * n for kind in (BAN, PICK) for side in (BLUE, RED)
        }
        self.marg_kind: dict[str, list[float]] = {kind: [0.0] * n for kind in (BAN, PICK)}
        self.kind_total: dict[str, float] = {BAN: 0.0, PICK: 0.0}
        self.pairs: dict[str, dict[tuple[int, int], float]] = {
            rel: {} for rel in (REL_PICK_SAME, REL_PICK_CROSS, REL_BAN, REL_BAN_PICK)
        }
        # 位置先验：(step, side, kind) -> hero 计数；仅有序数据会填
        self.pos: dict[tuple[int, int, str], list[float]] = {}
        self.hero_slots: list[float] = [0.0] * n
        self.total_slots: float = 0.0
        self.n_games: int = 0
        self.n_ordered_games: int = 0
        self.fitted_from: dict = {}
        # predict() 内部的分路缺口缓存（见 _lane_fit）。
        # M4 每展开一个节点就要给 130 个候选打分，而 gap(picked) 与候选无关、
        # 只跟「这一方已选谁」有关；不缓存的话同一份 120 种排列要重算 130 遍。
        self._gap_memo: dict | None = None
        # 预编译的打分表（见 _build_dense）：边际 / 成对 / 分路矩阵。
        # fit 或 from_dict 之后必须置空，否则会拿着旧表打分。
        self._dense: dict | None = None

    # ================= 训练 =================
    def _bump(self, vec: list[float], hero: int, w: float = 1.0) -> None:
        vec[hero] += w

    def _bump_pair(self, rel: str, x: int, y: int, w: float = 1.0) -> None:
        if x == y:
            return
        key = (x, y) if x < y else (y, x)
        table = self.pairs[rel]
        table[key] = table.get(key, 0.0) + w

    def _fit_game(self, game: GameSample, ordered: Iterable[tuple[int, str, int, int]] | None) -> None:
        self.n_games += 1
        for side in (BLUE, RED):
            for kind in (BAN, PICK):
                for h in game.heroes_of(side, kind):
                    self._bump(self.marg[(kind, side)], h)
                    self._bump(self.marg_kind[kind], h)
                    self.kind_total[kind] += 1.0
                    self.hero_slots[h] += 1.0
                    self.total_slots += 1.0

        bans_all = list(game.bans[BLUE]) + list(game.bans[RED])
        picks_blue = list(game.picks[BLUE])
        picks_red = list(game.picks[RED])
        picks_all = picks_blue + picks_red

        for i in range(len(picks_blue)):
            for j in range(i + 1, len(picks_blue)):
                self._bump_pair(REL_PICK_SAME, picks_blue[i], picks_blue[j])
        for i in range(len(picks_red)):
            for j in range(i + 1, len(picks_red)):
                self._bump_pair(REL_PICK_SAME, picks_red[i], picks_red[j])
        for x in picks_blue:
            for y in picks_red:
                self._bump_pair(REL_PICK_CROSS, x, y)
        for i in range(len(bans_all)):
            for j in range(i + 1, len(bans_all)):
                self._bump_pair(REL_BAN, bans_all[i], bans_all[j])
        for x in bans_all:
            for y in picks_all:
                self._bump_pair(REL_BAN_PICK, x, y)

        # 位置先验只有在真实手顺序下才有意义
        if ordered:
            self.n_ordered_games += 1
            for step_index, kind, side, hero in ordered:
                vec = self.pos.get((step_index, side, kind))
                if vec is None:
                    vec = [0.0] * self.pool.size
                    self.pos[(step_index, side, kind)] = vec
                self._bump(vec, hero)

    def fit(self, dataset: Dataset, template: BPTemplate | None = None) -> "TransitionModel":
        self._dense = None          # 表变了，预编译结果作废
        tpl = template or BPTemplate.load(dataset.template_id)
        for s in dataset.series:
            for g in s.games:
                ordered = None
                if dataset.order_known and g.order_known:
                    ordered = dataset.slot_plan(g, tpl)
                self._fit_game(g, ordered)
        self.template_id = tpl.id
        self.fitted_from = dataset.stats()
        return self

    # ================= 打分 =================
    def _log_marginal(self, kind: str, side: int, hero: int) -> float:
        cfg = self.config
        vec = self.marg[(kind, side)]
        total = self.kind_total[kind] / 2.0          # 每方各自的样本量
        n = (vec[hero] + cfg.alpha_marginal) / (total + cfg.alpha_marginal * self.pool.size)
        return math.log(max(n, 1e-12))

    def _log_lift(self, rel: str, x: int, y: int) -> float:
        """smoothed log-lift：实际共现次数 / 独立假设下的期望次数。"""
        cfg = self.config
        table = self.pairs[rel]
        key = (x, y) if x < y else (y, x)
        n_xy = table.get(key, 0.0)
        n_x = self.hero_slots[x]
        n_y = self.hero_slots[y]
        n_eff = max(self.total_slots, 1.0)
        expected = n_x * n_y / n_eff
        lift = (n_xy + cfg.alpha_pair) / (expected + cfg.alpha_pair)
        return max(-cfg.clip_log_lift, min(cfg.clip_log_lift, math.log(max(lift, 1e-6))))

    def _mean_lift(self, rel: str, x: int, context: list[int]) -> float:
        if not context:
            return 0.0
        return sum(self._log_lift(rel, x, y) for y in context) / len(context)

    def _lane_fit(self, state: DraftState, side: int, hero: int) -> float:
        """分路缺口：这一手能补上「还空着的路」的期望概率，∈[0,1]。

        旧版本把候选英雄的所有可打路都算成加分（摇摆位得 2 分），
        等于奖励「这个人能打两个位置」而不是「他补上了那个空位」，
        而且会让一个摇摆英雄同时填掉两条路。现在改成
        Σ_lane p(lane|hero) · P(lane 仍空着)，其中 P(lane 空着) 由
        当前选人的单射分配后验算出（见 lane.py）。摇摆英雄自然被折价。

        gap 只依赖「这一方已选谁」，与候选英雄无关 —— predict() 里会把它算一次
        放进 _gap_memo，供当次全部候选复用（M4 每个节点省 129 次 5! 枚举）。
        """
        lm = self.pool.lane_model()
        picked = state.picks_of(side)
        gaps = None
        if self._gap_memo is not None:
            key = (side, tuple(picked))
            gaps = self._gap_memo.get(key)
            if gaps is None:
                gaps = lm.gap(picked)
                self._gap_memo[key] = gaps
        if gaps is None:
            return lm.fill_value(picked, hero)
        return sum(p * g for p, g in zip(lm.probs(hero), gaps))

    def _log_position(self, state: DraftState, kind: str, side: int, hero: int) -> float | None:
        key = (state.step_index, side, kind)
        vec = self.pos.get(key)
        if vec is None:
            return None
        total = sum(vec)
        if total <= 0:
            return None
        cfg = self.config
        p = (vec[hero] + cfg.alpha_marginal) / (total + cfg.alpha_marginal * self.pool.size)
        return math.log(max(p, 1e-12))

    def score(self, state: DraftState, hero: int, kind: str | None = None,
              side: int | None = None) -> float:
        cfg = self.config
        kind = kind or state.kind_to_act or PICK
        side = side if side is not None else (state.side_to_act or state.our_side)
        opp = other(side)

        s = cfg.w_marginal * self._log_marginal(kind, side, hero)

        self_picks = state.picks_of(side)
        opp_picks = state.picks_of(opp)
        bans = state.bans()

        if kind == PICK:
            s += cfg.w_synergy * self._mean_lift(REL_PICK_SAME, hero, self_picks)
            s += cfg.w_counter * self._mean_lift(REL_PICK_CROSS, hero, opp_picks)
            s += cfg.w_ban_pick * self._mean_lift(REL_BAN_PICK, hero, bans)
            s += cfg.w_lane * self._lane_fit(state, side, hero)
        else:
            s += cfg.w_ban_pick * self._mean_lift(REL_BAN_PICK, hero, self_picks + opp_picks)
            s += cfg.w_ban * self._mean_lift(REL_BAN, hero, bans)

        pos = self._log_position(state, kind, side, hero)
        if pos is not None:
            s += cfg.w_position * pos
        return s

    # ================= 预测 =================
    def _build_dense(self) -> None:
        """把逐英雄打分要用的表预编译成 numpy 数组。

        为什么值得：`score()` 每评一个候选要做 6~7 项，其中成对项是「对每个已选英雄
        查一次表再取平均」——130 个候选 × 每次 6 个上下文 ≈ 800 次字典查找，单次
        predict 2.1ms。而 M4 的叶子补完要连着走 20 手 predict，一个叶子 40ms，
        深度 3 的 176 个叶子就是 7 秒，这就是「用真 M2 当补完策略」一直没落地的原因。

        预编译之后，同一项变成「整列索引 + mean(axis=1)」，一次 predict 降到 0.2ms 量级。
        代价是每次 load 多花约 0.1s、多占 4×N² 个 float（N=134 时约 0.6MB）——一次性。

        数值必须与 `score()` 逐项等价，否则 M2 的概率会整体漂移，
        M2 的命中率、M4 的推荐都会跟着变而没人知道为什么。
        由 tests/test_markov.py::test_vectorized_matches_scalar 守住。
        """
        import numpy as np

        cfg = self.config
        n = self.pool.size
        marg: dict[tuple[str, int], object] = {}
        for (kind, side), vec in self.marg.items():
            total = self.kind_total[kind] / 2.0        # 每方各自的样本量
            denom = total + cfg.alpha_marginal * n
            arr = (np.asarray(vec, dtype=float) + cfg.alpha_marginal) / denom
            marg[(kind, side)] = np.log(np.maximum(arr, 1e-12))

        # 成对 lift 的「共现为 0」底线：expected = n_x·n_y / n_eff
        n_x = np.asarray(self.hero_slots, dtype=float)
        n_eff = max(self.total_slots, 1.0)
        expected = np.outer(n_x, n_x) / n_eff
        floor = np.log(np.maximum(cfg.alpha_pair / (expected + cfg.alpha_pair), 1e-6))
        floor = np.clip(floor, -cfg.clip_log_lift, cfg.clip_log_lift)
        pair: dict[str, object] = {}
        for rel, table in self.pairs.items():
            mat = floor.copy()
            if table:
                keys = list(table.keys())
                rows = np.fromiter((a for a, _ in keys), dtype=int, count=len(keys))
                cols = np.fromiter((b for _, b in keys), dtype=int, count=len(keys))
                n_xy = np.fromiter((table[k] for k in keys), dtype=float, count=len(keys))
                lift = (n_xy + cfg.alpha_pair) / (expected[rows, cols] + cfg.alpha_pair)
                val = np.clip(np.log(np.maximum(lift, 1e-6)),
                              -cfg.clip_log_lift, cfg.clip_log_lift)
                mat[rows, cols] = val
                mat[cols, rows] = val        # 表是按 (小, 大) 存的，两边都要写
            pair[rel] = mat

        lm = self.pool.lane_model()
        lane = np.asarray([lm.probs(h) for h in range(n)], dtype=float)
        self._dense = {"marg": marg, "pair": pair, "lane": lane, "pos": {}}

    def _pos_vec(self, step_index: int, side: int, kind: str):
        """位置先验的 log 概率向量；该 (步序, 方, 禁/选) 没有样本时返回 None。"""
        import numpy as np

        cache = self._dense["pos"]
        key = (step_index, side, kind)
        if key in cache:
            return cache[key]
        vec = self.pos.get(key)
        if vec is None or sum(vec) <= 0:
            cache[key] = None
            return None
        cfg = self.config
        n = self.pool.size
        arr = (np.asarray(vec, dtype=float) + cfg.alpha_marginal) / (
            sum(vec) + cfg.alpha_marginal * n)
        out = np.log(np.maximum(arr, 1e-12))
        cache[key] = out
        return out

    def _score_all(self, state: DraftState, kind: str, side: int):
        """一次算出**全部**英雄的分数（逐项与 `score()` 对齐）。"""
        import numpy as np

        if self._dense is None:
            self._build_dense()
        cfg = self.config
        d = self._dense
        s = cfg.w_marginal * d["marg"][(kind, side)]
        self_picks = state.picks_of(side)
        opp_picks = state.picks_of(other(side))
        bans = state.bans()
        pair = d["pair"]
        if kind == PICK:
            if self_picks:
                s = s + cfg.w_synergy * pair[REL_PICK_SAME][:, self_picks].mean(axis=1)
            if opp_picks:
                s = s + cfg.w_counter * pair[REL_PICK_CROSS][:, opp_picks].mean(axis=1)
            if bans:
                s = s + cfg.w_ban_pick * pair[REL_BAN_PICK][:, bans].mean(axis=1)
            gaps = np.asarray(self.pool.lane_model().gap(self_picks), dtype=float)
            s = s + cfg.w_lane * (d["lane"] @ gaps)
        else:
            ctx = self_picks + opp_picks
            if ctx:
                s = s + cfg.w_ban_pick * pair[REL_BAN_PICK][:, ctx].mean(axis=1)
            if bans:
                s = s + cfg.w_ban * pair[REL_BAN][:, bans].mean(axis=1)
        pos = self._pos_vec(state.step_index, side, kind)
        if pos is not None:
            s = s + cfg.w_position * pos
        return s

    def predict(
        self,
        state: DraftState,
        top_k: int = 3,
        side: int | None = None,
        exclude_wasted: bool = False,
    ) -> Prediction:
        """给出下一手的分布。

        exclude_wasted：是否把「对方已用过、ban 了也空过」的英雄从候选里去掉。
        默认 False —— 预测对手行为时应保留（对手偶尔真的会空 ban），
        M4 搜索时传 True 以减少无意义分支。
        """
        kind = state.kind_to_act or PICK
        side = side if side is not None else (state.side_to_act or state.our_side)
        legal = state.legal_hero_ids(exclude_wasted=exclude_wasted)
        if not legal:
            return Prediction([], [], 0.0, 0, state.step_index, kind, side)

        cfg = self.config
        scores = self._score_all(state, kind, side)
        sc = [float(scores[h]) for h in legal]
        m = max(sc)
        exps = [(h, math.exp((s - m) / max(cfg.temperature, 1e-6)))
                for h, s in zip(legal, sc)]
        z = sum(e for _, e in exps)
        ranked = sorted(((h, e / z) for h, e in exps), key=lambda kv: (-kv[1], kv[0]))
        entropy = -sum(p * math.log(p) for _, p in ranked if p > 0)
        return Prediction(
            ranked=ranked,
            top_k=ranked[:top_k],
            entropy=entropy,
            n_legal=len(ranked),
            step_index=state.step_index,
            kind=kind,
            side=side,
        )

    # ================= 持久化 =================
    def to_dict(self) -> dict:
        return {
            "schema": 1,
            "template": self.template_id,
            "pool_patch": self.pool.patch,
            "pool_size": self.pool.size,
            "config": asdict(self.config),
            "n_games": self.n_games,
            "n_ordered_games": self.n_ordered_games,
            "fitted_from": self.fitted_from,
            "marg": {f"{k}|{side}": v for (k, side), v in self.marg.items()},
            "marg_kind": self.marg_kind,
            "kind_total": self.kind_total,
            "pairs": {rel: [[a, b, c] for (a, b), c in t.items()] for rel, t in self.pairs.items()},
            "pos": {f"{s}|{side}|{k}": v for (s, side, k), v in self.pos.items()},
            "hero_slots": self.hero_slots,
            "total_slots": self.total_slots,
        }

    @classmethod
    def from_dict(cls, data: dict, pool: HeroPool) -> "TransitionModel":
        if data.get("pool_size") != pool.size:
            raise ValueError(
                f"模型是按 {data.get('pool_size')} 个英雄训练的，当前池有 {pool.size} 个；"
                "英雄 id 错位会让预测完全失效，请重新训练"
            )
        m = cls(pool, MarkovConfig(**data["config"]), data.get("template"))
        m.marg = {}
        for key, vec in data["marg"].items():
            k, side = key.split("|")
            m.marg[(k, int(side))] = list(vec)
        m.marg_kind = {k: list(v) for k, v in data["marg_kind"].items()}
        m.kind_total = dict(data["kind_total"])
        m.pairs = {
            rel: {(int(a), int(b)): float(c) for a, b, c in triples}
            for rel, triples in data["pairs"].items()
        }
        m.pos = {}
        for key, vec in data["pos"].items():
            s, side, k = key.split("|")
            m.pos[(int(s), int(side), k)] = list(vec)
        m.hero_slots = list(data["hero_slots"])
        m.total_slots = float(data["total_slots"])
        m.n_games = int(data["n_games"])
        m.n_ordered_games = int(data.get("n_ordered_games", 0))
        m.fitted_from = data.get("fitted_from", {})
        return m

    def save(self, path: str | pathlib.Path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | pathlib.Path, pool: HeroPool | None = None) -> "TransitionModel":
        pool = pool or HeroPool.load()
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, pool)

    def __repr__(self) -> str:
        return (
            f"<TransitionModel games={self.n_games} ordered={self.n_ordered_games} "
            f"pairs={sum(len(t) for t in self.pairs.values())}>"
        )


# ====================== 评估 ======================
@dataclass
class EvalResult:
    n: int
    top1: float
    top3: float
    mrr: float
    nll: float
    uniform_nll: float
    by_kind: dict = field(default_factory=dict)
    replay_order: str = "canonical"

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "top1": round(self.top1, 4),
            "top3": round(self.top3, 4),
            "mrr": round(self.mrr, 4),
            "nll": round(self.nll, 4),
            "uniform_nll": round(self.uniform_nll, 4),
            "by_kind": self.by_kind,
            "replay_order": self.replay_order,
        }

    def __str__(self) -> str:
        return (
            f"n={self.n} top1={self.top1:.1%} top3={self.top3:.1%} "
            f"MRR={self.mrr:.3f} NLL={self.nll:.3f}（均匀基线 {self.uniform_nll:.3f}）"
        )


def evaluate(
    model: TransitionModel,
    dataset: Dataset,
    template: BPTemplate,
    top_k: int = 3,
) -> EvalResult:
    """逐步评估：每一步只看到它之前的局面，预测真实那一手。

    注意 replay 的顺序语义（见 dataset 模块说明）：
    数据只有集合时是 canonical 顺序，指标是**上界偏乐观**的代理值；
    有真实逐手数据时才是严格的 in-game 指标。
    """
    hits1 = hits3 = 0
    rr = 0.0
    nll = 0.0
    uni = 0.0
    n = 0
    per_kind: dict[str, list] = {BAN: [0, 0, 0, 0.0, 0.0], PICK: [0, 0, 0, 0.0, 0.0]}

    for _, _, steps in dataset.iter_replays(template):
        for st in steps:
            pred = model.predict(st.state, top_k=top_k, side=st.side)
            rank = pred.rank_of(st.hero_id)
            p = pred.prob_of(st.hero_id)
            n += 1
            if rank == 1:
                hits1 += 1
                per_kind[st.kind][1] += 1
            if rank is not None and rank <= top_k:
                hits3 += 1
                per_kind[st.kind][2] += 1
            rr += (1.0 / rank) if rank else 0.0
            nll += -math.log(max(p, 1e-12))
            uni += math.log(max(pred.n_legal, 1))
            per_kind[st.kind][0] += 1
            per_kind[st.kind][3] += (1.0 / rank) if rank else 0.0
            per_kind[st.kind][4] += -math.log(max(p, 1e-12))

    by_kind = {}
    for k, (cnt, h1, h3, r, l) in per_kind.items():
        if cnt:
            by_kind[k] = {
                "n": cnt,
                "top1": round(h1 / cnt, 4),
                "top3": round(h3 / cnt, 4),
                "mrr": round(r / cnt, 4),
                "nll": round(l / cnt, 4),
            }
    return EvalResult(
        n=n,
        top1=hits1 / n if n else 0.0,
        top3=hits3 / n if n else 0.0,
        mrr=rr / n if n else 0.0,
        nll=nll / n if n else 0.0,
        uniform_nll=uni / n if n else 0.0,
        by_kind=by_kind,
        replay_order="ordered" if dataset.order_known else "canonical",
    )
