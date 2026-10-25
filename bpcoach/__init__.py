# -*- coding: utf-8 -*-
"""BPCoach 后端 —— M1 局面编码、M2 选人预测、M3 胜率评估、M4 阵容推荐。

    M1 局面编码   state.py / series.py / template.py / lane.py
                  把一局 B/P 变成定长向量 + 合法动作集合 + 可哈希状态键；
                  摇摆位（马超可对抗可打野）按概率分布 + 单射分配处理
    M2 选人预测   dataset.py / markov.py
                  分层回退马尔可夫链 + 成对提升，输出对手下一手的前 k 个候选及概率
    M3 胜率评估   winrate.py / mvp.py
                  给定双方十手阵容输出胜率；额外的 MVP 预测头
    M4 阵容推荐   search.py
                  树搜索（默认等深全展开：我方取最大、对方按 M2 概率加权；UCT 保留）：
                  M2 当先验与剪枝、M3 当叶子估值，奖励折成系列赛晋级概率；
                  两个旋钮会改排序——「英雄池代价」与「对手口径 opp_risk」。
                  禁人步换一把尺子（拒绝价值），因为搜索值看不见「禁人」这件事

M1 的产出契约（M2/M3/M4 已全部用上）：

    state.encode()           -> list[int]   长度 N，0 未涉及 / -1 被 ban / 1 我方已选 / 2 敌方已选
    state.encode_pair()      -> (我方 0/1 向量, 敌方 0/1 向量)   论文里 110+110 的拼法
    state.available_mask()   -> list[int]   0/1，含全局 B/P 锁定（按队伍算）
    state.legal_hero_ids()   -> list[int]   当前这一步真正可执行的动作集合
    state.lane_profile()     -> dict        双方各分路的**期望占用**（摇摆位按概率摊开）
    state.meta()             -> dict        步数、行动方、禁/选、系列赛进度等标量特征
    state.model_input()      -> dict        以上打包，供 M3 直接喂模型
    state.state_key()        -> tuple       可哈希，供 M4 做置换表去重

M2 的产出契约（M4 直接用）：

    model = TransitionModel.load("models/markov_kpl2026.json")
    pred  = model.predict(state, top_k=3)
    pred.top_k               -> [(hero_id, prob)]   MCTS 展开的高概率分支
    pred.prob_of(hero_id)    -> float               条件概率，可作为先验
    pred.entropy             -> float               分布越平，搜索宽度越该放宽

M3 的产出契约：

    m3 = WinRateModel.load("models/winrate_kpl2026_lineup.joblib")
    m3.predict_state(state)          -> WinPrediction(p_win, spread)
    m3.predict_lineup(我方五个, 敌方五个) -> WinPrediction
    m3.explain(state)                -> [(英雄名, 拿掉他胜率掉多少)]

M4 的产出契约：

    search = DraftSearch.load("models")             # 或 search.load(...)
    rec = search.recommend(state, depth=3)          # -> Recommendation
    rec.best            -> Candidate(hero_id, value, p_win, p_greedy, equity, lane, ...)
    rec.candidates      -> 备选，按价值排序（对方回合时按「最伤我方」排序）
    rec.pv              -> 最优路径逐手（树认为的最优/最可能，不是预测）
    rec.replies         -> 主推荐之后对手的几种应法，以及我方 p 变成多少
    rec.baseline_p      -> M2 贪心走完的胜率（参照点）
    rec.rank_by         -> "value"（搜索值）| "denial"（禁人步的拒绝价值）
    rec.opp_risk        -> 对手口径：0 = 按 M2 概率加权（默认），1 = 最坏情况
    rec.degenerate      -> 所有候选同值（ban 步常见），此时「并列第一」不是推荐
    myopic = search.myopic(state)                   # 贪心基线，用来对比「搜索值不值」

单文件推理包（交付用）：

    from bpcoach import build_bundle, load_bundle
    build_bundle()                                  # -> dist/bpcoach_v<版本>.bpc
    search = load_bundle("dist/bpcoach_v0.5.0.bpc") # 校验 sha256 后一行装好
"""

from .bundle import build_bundle, load_bundle, read_manifest
from .dataset import Dataset, GameSample, SeriesSample
from .errors import BPCoachError, IllegalActionError, UnknownHeroError
from .heroes import Hero, HeroPool
from .lane import LaneModel
from .markov import MarkovConfig, Prediction, TransitionModel, evaluate
from .search import (
    Candidate, DraftSearch, Recommendation, RewardConfig, SearchConfig,
    clinch_probability, expected_wins, series_equity, series_swing, strength_offset,
)
from .series import SeriesContext
from .state import DraftState
from .template import BPTemplate, Step
from .winrate import HeroPriors, WinEval, WinPrediction, WinRateModel, loso_evaluate

__all__ = [
    "BPCoachError",
    "IllegalActionError",
    "UnknownHeroError",
    "Hero",
    "HeroPool",
    "LaneModel",
    "SeriesContext",
    "DraftState",
    "BPTemplate",
    "Step",
    "Dataset",
    "GameSample",
    "SeriesSample",
    "MarkovConfig",
    "TransitionModel",
    "Prediction",
    "evaluate",
    "HeroPriors",
    "WinRateModel",
    "WinPrediction",
    "WinEval",
    "loso_evaluate",
    "DraftSearch",
    "SearchConfig",
    "RewardConfig",
    "Candidate",
    "Recommendation",
    "clinch_probability",
    "expected_wins",
    "series_equity",
    "series_swing",
    "strength_offset",
    "build_bundle",
    "load_bundle",
    "read_manifest",
]

__version__ = "0.5.0"
