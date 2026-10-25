# -*- coding: utf-8 -*-
"""M3 胜率评估：给定双方阵容，估「我方赢下这一局」的概率。

论文里的做法
    把 110 维局面向量（我方 10 个 pick + 敌方 10 个 pick 的 one-hot）喂给随机森林，
    1182 局训练，得到 Acc 0.564 / AUC 0.572。作者自己也承认这个数不高，
    但比「看阵容感觉」稳定，且能给出概率供 MCTS 使用。

这里的做法与三处必要的偏离
    1. **特征维度要跟着样本量走**。现在有 1447 局真实标签（347 个系列赛），
       但 2N = 268 维的 one-hot 依然偏多，实测反而不如 7 维差分（见 README 的配置表）。
       所以提供四种特征集：
         paper     —— 2N 维 one-hot，完全复刻论文
         compact   —— 17 维手工特征（先验胜率/先验 MVP/分路似然/拥挤度/摇摆数/上下文）
         diff      —— 7 维差分，只看阵容时的最佳可解释口径
         bt        —— one-hot 差分（Bradley-Terry 式），N 个系数，只看阵容时 AUC 最高
       特征集是配置项，上层接口不变。
    2. **评测口径**。必须按系列赛分组做留一（leave-one-series-out），
       否则同一场 BO5 的 5 局会同时出现在训练集和测试集里，
       同队同版本同一天的 B/P 高度相关，指标会虚高得离谱。
    3. **英雄级先验要做收缩**。某个英雄 3 场 3 胜不能说明它强，
       用 (wins + k*base) / (n + k) 把它拉回基准率，n 小的时候自动不生效。

一个必须说清楚的天花板
    B/P 只能解释胜负的一小部分——同一套阵容换个选手、换个状态就可能是另一个结果。
    论文 0.56 的准确率已经接近「阵容信息的天花板」。
    所以 M3 的输出定位是**给 M4 提供相对排序与概率**，而不是预测比赛结果。
    评估脚本会同时报「均匀基线 / 只看先手方」两个基线，方便判断模型是否真的学到了东西。
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

from .dataset import Dataset, GameSample, SeriesSample
from .heroes import HeroPool
from .series import BLUE, RED
from .state import DraftState
from .template import BAN, PICK

FEATURE_SETS = ("paper", "compact", "enhanced", "diff", "bt", "bt_ctx")
# 收缩强度：相当于先验占 8 局的权重（职业赛一个英雄一季也就几十次出场）
SHRINK = 8.0


# ====================== 英雄先验 ======================
@dataclass
class HeroPriors:
    """英雄级先验。全部带收缩，缺数据时自动退回基准率。"""

    win: list[float] = field(default_factory=list)
    mvp: list[float] = field(default_factory=list)
    pick: list[float] = field(default_factory=list)
    ban: list[float] = field(default_factory=list)
    # 队伍强度：名称 -> 收缩后的胜率。这是「赛前就知道」的信息，
    # 与阵容无关，所以单独放一列、单独报告，不能混进「阵容有多强」的解释里。
    team: dict[str, float] = field(default_factory=dict)
    base_win: float = 0.5
    n_games: int = 0
    n_mvp: int = 0

    @classmethod
    def fit(cls, dataset: Dataset) -> "HeroPriors":
        labeled = dataset.labeled_games()
        n = dataset.pool.size
        win_c = [0.0] * n
        app_c = [0.0] * n
        mvp_c = [0.0] * n
        pick_c = [0.0] * n
        ban_c = [0.0] * n
        wins = 0
        for _, g in labeled:
            if g.winner == "blue":
                wins += 1
                win_side, lose_side = BLUE, RED
            else:
                win_side, lose_side = RED, BLUE
            for h in g.picks[win_side]:
                win_c[h] += 1
                app_c[h] += 1
            for h in g.picks[lose_side]:
                app_c[h] += 1
            for side in (BLUE, RED):
                for h in g.bans[side]:
                    ban_c[h] += 1
            if g.mvp_hero is not None:
                mvp_c[g.mvp_hero] += 1
        for _, g in dataset.labeled_games():
            for side in (BLUE, RED):
                for h in g.picks[side]:
                    pick_c[h] += 1

        base = wins / len(labeled) if labeled else 0.5
        mvp_base = sum(mvp_c) / max(len(labeled), 1)
        # 队伍强度：alpha=12 的收缩。一支队一季 20~40 局，
        # 不收缩的话「3 战 3 胜」的队会被估成神仙。
        t_win: dict[str, float] = {}
        t_n: dict[str, float] = {}
        for _, g in labeled:
            for name, won in ((g.blue, g.winner == "blue"), (g.red, g.winner == "red")):
                t_n[name] = t_n.get(name, 0.0) + 1.0
                t_win[name] = t_win.get(name, 0.0) + (1.0 if won else 0.0)
        team = {k: (t_win[k] + 12 * base) / (t_n[k] + 12) for k in t_n}
        win_p = [(win_c[i] + SHRINK * base) / (app_c[i] + SHRINK) for i in range(n)]
        mvp_p = [(mvp_c[i] + SHRINK * mvp_base) / (app_c[i] + SHRINK) for i in range(n)]
        tot_pick = sum(pick_c) or 1.0
        tot_ban = sum(ban_c) or 1.0
        return cls(
            win=win_p, mvp=mvp_p,
            pick=[c / tot_pick for c in pick_c],
            ban=[c / tot_ban for c in ban_c],
            team=team,
            base_win=base,
            n_games=len(labeled),
            n_mvp=int(sum(mvp_c)),
        )

    def mean(self, table: list[float], hero_ids: tuple[int, ...] | list[int]) -> float:
        if not hero_ids:
            return 1.0 / max(len(table), 1) if table else 0.0
        return sum(table[h] for h in hero_ids) / len(hero_ids)

    def to_dict(self) -> dict:
        return {
            "base_win": round(self.base_win, 4),
            "n_games": self.n_games,
            "n_mvp": self.n_mvp,
            "top_win": sorted(range(len(self.win)), key=lambda i: -self.win[i])[:10],
        }


# ====================== 特征装配 ======================
PROFILE_DIMS = ("prior_win", "prior_mvp", "lanes_ll", "neg_crowd", "flex")


def team_profile(pool: HeroPool, priors: HeroPriors, picks: list[int]) -> list[float]:
    """单边阵容画像（5 维）。差分模型用它做 φ(我方) − φ(敌方)。

    为什么单独抽出来：B/P 的胜负本质是**相对**的，绝对水平没意义。
    把一边压缩成 5 维画像，再在差空间里学权重，参数少、可解释、
    而且天然满足「交换两队则胜率取补」这条对称性——n=26 的时候这点很重要。
    """
    lm = pool.lane_model()
    occ = lm.occupancy(picks) if picks else [0.0] * len(lm.lanes)
    crowd = sum(max(0.0, o - 1.0) for o in occ)
    flex = sum(1 for h in picks if lm.is_flex(h))
    return [
        priors.mean(priors.win, picks),
        priors.mean(priors.mvp, picks),
        lm.log_assignment(picks) if picks else 0.0,
        -crowd,
        flex / 5.0,
    ]


def lineup_features(
    pool: HeroPool,
    priors: HeroPriors,
    our_picks: list[int],
    opp_picks: list[int],
    is_blue: bool = True,
    game_index: int = 1,
    score_diff: int = 0,
    total_bans: int = 0,
    feature_set: str = "compact",
) -> tuple[list[float], list[str]]:
    """把「我方五个 + 敌方五个」整理成模型输入。

    compact 的 14 个特征都对应一条职业赛常识，不是为了凑维度：
        先验胜率差 —— 版本强势英雄的集合差异
        MVP 率差   —— carry 能力（有 MVP 数据时才有效，缺数据自动退化为 0 方差）
        分路似然差 —— 这五个人能不能排成合理的 1-1-1-1-1
        拥挤度差   —— 有没有两个人都只能打同一路（阵容畸形）
        摇摆数差   —— 摇摆位多的阵容更抗针对
        先手方     —— 蓝方在 2026 年总有明显优势（数据里 65%）
        局号/比分  —— 打到第几局、大比分如何（英雄池消耗与心理压力）
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"未知特征集 {feature_set}，可选 {FEATURE_SETS}")
    lm = pool.lane_model()

    names: list[str] = []
    vec: list[float] = []

    if feature_set in ("paper", "enhanced"):
        ours = [0.0] * pool.size
        opp = [0.0] * pool.size
        for h in our_picks:
            ours[h] = 1.0
        for h in opp_picks:
            opp[h] = 1.0
        vec += ours + opp
        names += [f"ours:{h.name}" for h in pool] + [f"opp:{h.name}" for h in pool]

    our_win = priors.mean(priors.win, our_picks)
    opp_win = priors.mean(priors.win, opp_picks)
    our_mvp = priors.mean(priors.mvp, our_picks)
    opp_mvp = priors.mean(priors.mvp, opp_picks)
    our_ll = lm.log_assignment(our_picks) if our_picks else 0.0
    opp_ll = lm.log_assignment(opp_picks) if opp_picks else 0.0
    our_occ = lm.occupancy(our_picks) if our_picks else [0.0] * len(lm.lanes)
    opp_occ = lm.occupancy(opp_picks) if opp_picks else [0.0] * len(lm.lanes)
    our_crowd = sum(max(0.0, o - 1.0) for o in our_occ)
    opp_crowd = sum(max(0.0, o - 1.0) for o in opp_occ)
    our_flex = sum(1 for h in our_picks if lm.is_flex(h))
    opp_flex = sum(1 for h in opp_picks if lm.is_flex(h))

    compact = [
        our_win, opp_win, our_win - opp_win,
        our_mvp, opp_mvp, our_mvp - opp_mvp,
        our_ll, opp_ll, our_ll - opp_ll,
        our_crowd, opp_crowd, our_crowd - opp_crowd,
        (our_flex - opp_flex) / 5.0,
        1.0 if is_blue else 0.0,
        game_index / 5.0,
        score_diff / 3.0,
        total_bans / 20.0,
    ]
    compact_names = [
        "prior_win_ours", "prior_win_opp", "prior_win_diff",
        "prior_mvp_ours", "prior_mvp_opp", "prior_mvp_diff",
        "lanes_ll_ours", "lanes_ll_opp", "lanes_ll_diff",
        "crowd_ours", "crowd_opp", "crowd_diff",
        "flex_diff", "is_blue", "game_index", "score_diff", "ban_progress",
    ]
    vec += compact
    names += compact_names
    return vec, names


def game_features(dataset: Dataset, priors: HeroPriors, series: SeriesSample,
                  game: GameSample, feature_set: str = "compact"):
    """从一局真实数据里取特征。series 上下文用于「打之前的大比分」。"""
    before = [g for g in series.games if g.game_index < game.game_index]
    score_diff = sum(
        (1 if g.winner == "blue" else -1) for g in before if g.winner in ("blue", "red")
    )
    total_bans = len(game.bans[BLUE]) + len(game.bans[RED])
    return lineup_features(
        dataset.pool, priors,
        our_picks=list(game.picks[BLUE]),
        opp_picks=list(game.picks[RED]),
        is_blue=True,
        game_index=game.game_index,
        score_diff=score_diff,
        total_bans=total_bans,
        feature_set=feature_set,
    )


# ====================== 模型 ======================
@dataclass
class WinPrediction:
    p_win: float          # 我方（our_picks 一方）胜率
    spread: float         # 森林各树预测的标准差，作为不确定性
    n_trees: int
    feature_set: str

    def __str__(self) -> str:
        return f"胜率 {self.p_win:.1%}（±{self.spread:.1%}）"


class WinRateModel:
    """随机森林胜率模型。fit 之后可保存/加载，predict 面向 M4。"""

    def __init__(
        self,
        pool: HeroPool,
        feature_set: str = "compact",
        kind: str = "rf",
        seed: int = 42,
        n_estimators: int = 400,
        priors: HeroPriors | None = None,
        use_team: bool = False,
    ) -> None:
        self.pool = pool
        self.use_team = use_team
        self.feature_set = "diff" if kind == "diff_lr" else feature_set
        self.kind = kind
        self.seed = seed
        self.n_estimators = n_estimators
        self.priors = priors
        self.model = None
        self.feature_names: list[str] = []
        # 常数列（例如没有 MVP 数据时的 prior_mvp）在拟合前剔除：
        # 留着只会产生 0 系数和共线性，还会让「这是重要特征」的误读出现
        self.keep: list[int] | None = None
        self.meta: dict = {}

    # ---------------- 训练 ----------------
    def _new_model(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        if self.kind == "rf":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=6, min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=self.seed, n_jobs=1,
            )
        if self.kind == "gbdt":
            return GradientBoostingClassifier(random_state=self.seed, max_depth=2)
        if self.kind in ("lr", "diff_lr"):
            return LogisticRegression(max_iter=2000, C=0.5)
        raise ValueError(f"未知模型 {self.kind}")

    def _design_diff(self, picks_blue: list[int], picks_red: list[int],
                     game_index: int, score_diff: int) -> list[float]:
        """差分设计矩阵：φ(蓝) − φ(红) + 上下文。

        解释方式：系数 w 回答「这个特征每比对手多 1，蓝方胜率的对数几率变化多少」。
        截距单独承担「蓝方基本优势」，与阵容无关，M4 比较候选阵容时不会被它带偏。
        """
        pb = team_profile(self.pool, self.priors, picks_blue)
        pr = team_profile(self.pool, self.priors, picks_red)
        return [a - b for a, b in zip(pb, pr)] + [score_diff / 3.0, game_index / 5.0]

    @property
    def _diff_names(self) -> list[str]:
        return [f"d_{d}" for d in PROFILE_DIMS] + ["d_score_diff", "d_game_index"]

    def _xy(self, dataset: Dataset, rows: list[tuple[SeriesSample, GameSample]]):
        X, y = [], []
        if self.kind == "diff_lr":
            for s, g in rows:
                before = [x for x in s.games if x.game_index < g.game_index]
                sd = sum((1 if x.winner == "blue" else -1) for x in before
                         if x.winner in ("blue", "red"))
                X.append(self._design_diff(list(g.picks[BLUE]), list(g.picks[RED]),
                                           g.game_index, sd))
                y.append(1 if g.winner == "blue" else 0)
            self.feature_names = self._diff_names
            return X, y
        for s, g in rows:
            vec, names = game_features(dataset, self.priors, s, g, self.feature_set)
            X.append(vec)
            y.append(1 if g.winner == "blue" else 0)
        self.feature_names = names
        return X, y

    def fit(self, dataset: Dataset) -> "WinRateModel":
        rows = dataset.labeled_games()
        if len(rows) < 8:
            raise ValueError(
                f"只有 {len(rows)} 局带标签的对局，无法训练。"
                "先跑 tools/fetch_kpl_results.py 补标签。"
            )
        self.priors = self.priors or HeroPriors.fit(dataset)
        # 走特征缓存：与 loso_evaluate 完全同一条路径，
        # 避免「评估时用一套特征、训练时用另一套」这种最难发现的错位
        cache = FeatureCache(dataset, self.feature_set)
        X = cache.design(self.priors, self.use_team)
        y = cache.y
        self.feature_names = cache.names(self.use_team)
        if len(set(y)) < 2:
            raise ValueError("标签只有一类（全蓝胜或全红胜），检查 label_stats()")
        import numpy as np

        arr = np.asarray(X, dtype=float)
        std = arr.std(axis=0)
        keep = [i for i in range(arr.shape[1]) if std[i] > 1e-9]
        if not keep:            # 极端情况：所有特征都是常数，保留第一列让模型能跑
            keep = [0]
        self.keep = keep
        self.feature_names = [self.feature_names[i] for i in keep]
        self.model = self._new_model()
        self.model.fit(arr[:, keep], y)
        self.meta = {
            "n_games": len(rows),
            "n_series": len({s.id for s, _ in rows}),
            "feature_set": self.feature_set,
            "kind": self.kind,
            "seed": self.seed,
            "use_team": self.use_team,
            "base_win": round(float(sum(y) / len(y)), 4),
            "priors": self.priors.to_dict() if self.priors else {},
        }
        return self

    # ---------------- 预测 ----------------
    def _apply_keep(self, vec: list[float]) -> list[float]:
        return [vec[i] for i in self.keep] if self.keep is not None else vec

    def _proba(self, vec: list[float]) -> tuple[float, float]:
        import numpy as np

        X = np.asarray([[self._apply_keep(vec)]], dtype=float).reshape(1, -1)
        p = float(self.model.predict_proba(X)[0][1])
        if self.kind == "rf":
            tree_p = np.array([t.predict_proba(X)[0][1] for t in self.model.estimators_])
            spread = float(tree_p.std())
        else:
            spread = 0.0
        return p, spread

    def predict_lineup(
        self,
        our_picks: list[int],
        opp_picks: list[int],
        is_blue: bool = True,
        game_index: int = 1,
        score_diff: int = 0,
        total_bans: int = 0,
        team_blue: str | None = None,
        team_red: str | None = None,
    ) -> WinPrediction:
        if self.kind == "diff_lr":
            import numpy as np

            design = self._design_diff(
                list(our_picks) if is_blue else list(opp_picks),
                list(opp_picks) if is_blue else list(our_picks),
                game_index, score_diff,
            )
            if self.use_team:
                ts = self.priors.team or {}
                design = design + [ts.get(team_blue or "", 0.5) - ts.get(team_red or "", 0.5)]
            z = float(self.model.decision_function([self._apply_keep(design)])[0])       # 这是「蓝方胜」的对数几率
            p_blue = 1.0 / (1.0 + math.exp(-z))
            p = p_blue if is_blue else 1.0 - p_blue
            return WinPrediction(p, 0.0, 0, self.feature_set)

        vec, _ = lineup_features(
            self.pool, self.priors, our_picks, opp_picks,
            is_blue=is_blue, game_index=game_index, score_diff=score_diff,
            total_bans=total_bans, feature_set=self.feature_set,
        )
        if self.use_team:
            ts = self.priors.team or {}
            vec = vec + [ts.get(team_blue or "", 0.5) - ts.get(team_red or "", 0.5)]
        p, spread = self._proba(vec)
        return WinPrediction(p, spread, self.n_estimators if self.kind == "rf" else 0,
                             self.feature_set)

    def predict_state(self, state: DraftState, our_side: int | None = None) -> WinPrediction:
        """M4 的接口：传一个（通常是走完 20 手的）局面，返回我方胜率。"""
        side = our_side if our_side is not None else state.our_side
        opp = RED if side == BLUE else BLUE
        score = state.series.score() if state.series else {BLUE: 0, RED: 0}
        return self.predict_lineup(
            our_picks=state.picks_of(side),
            opp_picks=state.picks_of(opp),
            is_blue=(side == BLUE),
            game_index=state.game_index,
            score_diff=score.get(side, 0) - score.get(opp, 0),
            total_bans=len(state.bans()),
        )

    def explain(self, state: DraftState, top: int = 3) -> list[tuple[str, float]]:
        """粗略解释：把每个英雄从阵容里删掉，看胜率掉多少。

        这不是 SHAP，但对教练足够直观——「这手到底顶不顶用」。
        小样本下别把它当因果，只当相对重要度。
        """
        side = state.our_side
        opp = RED if side == BLUE else BLUE
        base = self.predict_state(state).p_win
        out: list[tuple[str, float]] = []
        for h in state.picks_of(side):
            rest = [x for x in state.picks_of(side) if x != h]
            p = self.predict_lineup(rest, state.picks_of(opp)).p_win
            out.append((self.pool.name_of(h), base - p))
        out.sort(key=lambda kv: -kv[1])
        return out[:top]

    # ---------------- 持久化 ----------------
    def save(self, path: str | pathlib.Path) -> None:
        import joblib

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_set": self.feature_set,
            "kind": self.kind,
            "seed": self.seed,
            "use_team": self.use_team,
            "pool_patch": self.pool.patch,
            "pool_size": self.pool.size,
            "priors": self.priors,
            "feature_names": self.feature_names,
            "keep": self.keep,
            "meta": self.meta,
            "template": None,
        }, p)
        meta_path = p.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "feature_set": self.feature_set, "kind": self.kind,
            "feature_names": self.feature_names, "meta": self.meta,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | pathlib.Path, pool: HeroPool | None = None) -> "WinRateModel":
        import joblib

        blob = joblib.load(path)
        return cls.from_blob(blob, pool)

    @classmethod
    def from_blob(cls, blob: dict, pool: HeroPool | None = None) -> "WinRateModel":
        """从 joblib 反序列化出的 dict 重建（bundle 加载用，与 load 同一份校验）。"""
        pool = pool or HeroPool.load()
        if blob.get("pool_size") != pool.size:
            raise ValueError(
                f"模型按 {blob.get('pool_size')} 个英雄训练，当前池 {pool.size} 个；英雄 id 会错位"
            )
        m = cls(pool, blob["feature_set"], blob["kind"], blob["seed"],
                priors=blob["priors"], use_team=bool(blob.get("use_team", False)))
        m.model = blob["model"]
        m.feature_names = blob["feature_names"]
        m.keep = blob.get("keep")
        m.meta = blob["meta"]
        return m

    def __repr__(self) -> str:
        return (f"<WinRateModel {self.kind}/{self.feature_set} "
                f"games={self.meta.get('n_games', 0)}>")


# ====================== 特征缓存 ======================
class FeatureCache:
    """把「与训练折无关」的特征一次算好，之后每折只重算依赖英雄先验的列。

    为什么需要它
        留一评估要对每个系列赛训练一次模型。原实现每折都把所有对局的
        `log_assignment()` 重跑一遍，而它内部要枚举 5! = 120 种分路排列：
        1447 局 × 2 边 × 120 排列 × 347 折 ≈ 1.2 亿次枚举，单次评估跑不完。
        实际上分路似然、拥挤度、摇摆数、局号这些**不随训练折变化**，
        依赖训练折的只有「英雄先验胜率 / 先验 MVP 率」两列，
        于是一次算好静态部分，每折只做两次 numpy 索引平均。
    """

    def __init__(self, dataset: Dataset, feature_set: str = "compact") -> None:
        self.feature_set = feature_set
        self.rows = dataset.labeled_games()
        lm = dataset.pool.lane_model()
        self.pool = dataset.pool
        self.teams: list[tuple[str, str]] = []

        self.picks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.ll: list[tuple[float, float]] = []
        self.crowd: list[tuple[float, float]] = []
        self.flex: list[tuple[float, float]] = []
        self.ctx: list[tuple[float, float, float]] = []      # gi/5, sd/3, bans/20
        self.y: list[int] = []
        self.onehot: list[list[float]] = []

        n = self.pool.size
        for s, g in self.rows:
            pb, pr = tuple(g.picks[BLUE]), tuple(g.picks[RED])
            self.picks.append((pb, pr))
            self.teams.append((g.blue, g.red))
            self.y.append(1 if g.winner == "blue" else 0)
            self.ll.append((lm.log_assignment(list(pb)), lm.log_assignment(list(pr))))
            occ_b, occ_r = lm.occupancy(list(pb)), lm.occupancy(list(pr))
            self.crowd.append((sum(max(0.0, o - 1.0) for o in occ_b),
                               sum(max(0.0, o - 1.0) for o in occ_r)))
            self.flex.append((float(sum(1 for h in pb if lm.is_flex(h))),
                              float(sum(1 for h in pr if lm.is_flex(h)))))
            before = [x for x in s.games if x.game_index < g.game_index]
            sd = sum((1 if x.winner == "blue" else -1) for x in before
                     if x.winner in ("blue", "red"))
            self.ctx.append((g.game_index / 5.0, sd / 3.0,
                             (len(g.bans[BLUE]) + len(g.bans[RED])) / 20.0))
            if feature_set in ("paper", "enhanced"):
                row = [0.0] * (2 * n)
                for h in pb:
                    row[h] = 1.0
                for h in pr:
                    row[n + h] = 1.0
                self.onehot.append(row)
            elif feature_set in ("bt", "bt_ctx"):
                row = [0.0] * n
                for h in pb:
                    row[h] += 1.0
                for h in pr:
                    row[h] -= 1.0
                self.onehot.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def design(self, priors: HeroPriors, use_team: bool = False) -> list[list[float]]:
        """给定英雄先验，产出设计矩阵。use_team 时追加「队伍强度差」。"""
        win, mvp = priors.win, priors.mvp
        out: list[list[float]] = []
        for i, (pb, pr) in enumerate(self.picks):
            ow = sum(win[h] for h in pb) / len(pb)
            opw = sum(win[h] for h in pr) / len(pr)
            om = sum(mvp[h] for h in pb) / len(pb)
            opm = sum(mvp[h] for h in pr) / len(pr)
            llb, llr = self.ll[i]
            cb, cr = self.crowd[i]
            fb, fr = self.flex[i]
            gi, sd, ban = self.ctx[i]
            if self.feature_set == "diff":
                row = [ow - opw, om - opm, llb - llr, -cb + cr, (fb - fr) / 5.0, sd, gi]
            elif self.feature_set in ("bt", "bt_ctx"):
                # 差分 one-hot：只学「这个英雄在蓝方比在红方多赢多少」一个系数，
                # 参数从 2N 降到 N，且天然满足「换边则概率取补」的对称性。
                row = list(self.onehot[i])
                if self.feature_set == "bt_ctx":
                    row += [sd, gi, ban]
            else:
                row = [ow, opw, ow - opw,
                       om, opm, om - opm,
                       llb, llr, llb - llr,
                       cb, cr, cb - cr,
                       (fb - fr) / 5.0, 1.0, gi, sd, ban]
                if self.feature_set in ("paper", "enhanced"):
                    row = self.onehot[i] + row     # 与 lineup_features 同口径：one-hot + compact17
            if use_team:
                ts = priors.team or {}
                row = row + [ts.get(self.teams[i][0], 0.5) - ts.get(self.teams[i][1], 0.5)]
            out.append(row)
        return out

    def names(self, use_team: bool = False) -> list[str]:
        n = self.pool.size
        if self.feature_set == "diff":
            base = ["d_prior_win", "d_prior_mvp", "d_lanes_ll", "d_neg_crowd",
                    "d_flex", "d_score_diff", "d_game_index"]
        elif self.feature_set in ("bt", "bt_ctx"):
            base = [f"d:{h.name}" for h in self.pool]
            if self.feature_set == "bt_ctx":
                base = base + ["score_diff", "game_index", "ban_progress"]
        elif self.feature_set in ("paper", "enhanced"):
            base = ([f"ours:{h.name}" for h in self.pool] +
                    [f"opp:{h.name}" for h in self.pool] +
                    ["prior_win_ours", "prior_win_opp", "prior_win_diff",
                     "prior_mvp_ours", "prior_mvp_opp", "prior_mvp_diff",
                     "lanes_ll_ours", "lanes_ll_opp", "lanes_ll_diff",
                     "crowd_ours", "crowd_opp", "crowd_diff",
                     "flex_diff", "is_blue", "game_index", "score_diff", "ban_progress"])
        else:
            base = ["prior_win_ours", "prior_win_opp", "prior_win_diff",
                    "prior_mvp_ours", "prior_mvp_opp", "prior_mvp_diff",
                    "lanes_ll_ours", "lanes_ll_opp", "lanes_ll_diff",
                    "crowd_ours", "crowd_opp", "crowd_diff",
                    "flex_diff", "is_blue", "game_index", "score_diff", "ban_progress"]
        if use_team:
            base = base + ["team_strength_diff"]
        return base


# ====================== 评估 ======================
@dataclass
class WinEval:
    n: int
    acc: float
    auc: float
    brier: float
    logloss: float
    acc_ci: tuple[float, float]
    auc_ci: tuple[float, float]
    baseline_acc: float
    folds: list[dict] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "acc": round(self.acc, 4), "auc": round(self.auc, 4),
            "brier": round(self.brier, 4), "logloss": round(self.logloss, 4),
            "acc_95ci": [round(x, 4) for x in self.acc_ci],
            "auc_95ci": [round(x, 4) for x in self.auc_ci],
            "baseline_acc": round(self.baseline_acc, 4),
            "folds": self.folds, "note": self.note,
        }

    def __str__(self) -> str:
        return (f"n={self.n} Acc={self.acc:.1%} [95%CI {self.acc_ci[0]:.1%}~{self.acc_ci[1]:.1%}] "
                f"AUC={self.auc:.3f} Brier={self.brier:.3f} LognLoss={self.logloss:.3f} "
                f"(多数类基线 {self.baseline_acc:.1%})")


def _bootstrap_ci(y: list[int], p: list[float], seed: int = 0, rounds: int = 1000):
    import random

    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)
    n = len(y)
    accs, aucs = [], []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        pp = [p[i] for i in idx]
        accs.append(sum(1 for a, b in zip(yy, pp) if (b >= 0.5) == (a == 1)) / n)
        aucs.append(roc_auc_score(yy, pp))
    accs.sort()
    aucs.sort()
    pick = lambda arr: (arr[int(0.025 * len(arr))], arr[int(0.975 * len(arr)) - 1])  # noqa: E731
    return pick(accs), pick(aucs)


def _make_estimator(kind: str, seed: int, n_estimators: int):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=6, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        )
    if kind == "gbdt":
        return GradientBoostingClassifier(random_state=seed, max_depth=2)
    if kind in ("lr", "diff_lr"):
        return LogisticRegression(max_iter=3000, C=0.5)
    raise ValueError(f"未知模型 {kind}")


def loso_evaluate(
    dataset: Dataset,
    factory=None,
    feature_set: str = "compact",
    kind: str = "rf",
    seed: int = 42,
    max_groups: int | None = None,
    n_estimators: int = 400,
    use_team: bool = False,
    collect: list | None = None,
) -> WinEval:
    """留一系列交叉验证：每次留出一个系列赛，其余训练。

    这是唯一诚实的口径。同一场 BO5 里 5 局共享队伍、版本、当天状态，
    随机切分会让模型「背下这场比赛」而指标虚高。

    训练折内的一切（英雄先验、队伍强度、模型权重）都只用训练折算，
    测试折的对局只出现在最后一次 predict 里。

    传 collect=[] 时，会把每局的 (系列赛 id, 局号, 预测胜率, 实际标签) 追加进去，
    便于演示脚本原样展示「留一预测 vs 实际」，而不是拿全量模型自夸。
    """
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    if kind == "diff_lr" and feature_set not in ("bt", "bt_ctx"):
        feature_set = "diff"
    cache = FeatureCache(dataset, feature_set)
    rows = cache.rows
    groups = sorted({s.id for s, _ in rows})
    if max_groups and len(groups) > max_groups:
        # 大样本时全留一太慢：等距抽样，仍保证按系列赛分组
        step = len(groups) / max_groups
        groups = [groups[int(i * step)] for i in range(max_groups)]

    y_all: list[int] = []
    p_all: list[float] = []
    folds: list[dict] = []

    for hold in groups:
        train_idx = [i for i, (s, _) in enumerate(rows) if s.id != hold]
        test_idx = [i for i, (s, _) in enumerate(rows) if s.id == hold]
        y_tr = [cache.y[i] for i in train_idx]
        if not test_idx or len(set(y_tr)) < 2:
            continue
        sub = Dataset(dataset.pool, [s for s in dataset.series if s.id != hold], dataset.meta)
        priors = HeroPriors.fit(sub)
        X = cache.design(priors, use_team)
        arr = np.asarray(X, dtype=float)
        model = _make_estimator(kind, seed, n_estimators)
        model.fit(arr[train_idx], y_tr)
        proba = model.predict_proba(arr[test_idx])[:, 1]
        y_all += [cache.y[i] for i in test_idx]
        p_all += [float(p) for p in proba]
        if collect is not None:
            for i, p in zip(test_idx, proba):
                s_, g_ = rows[i]
                collect.append((s_.id, g_.game_index, float(p), cache.y[i],
                                g_.blue, g_.red, s_.date))
        folds.append({"holdout": hold, "n_test": len(test_idx)})

    if not y_all:
        raise ValueError("没有足够的带标签系列赛做留一验证（至少需要 2 个系列赛、且标签不同类）")

    acc = sum(1 for a, b in zip(y_all, p_all) if (b >= 0.5) == (a == 1)) / len(y_all)
    auc = roc_auc_score(y_all, p_all) if len(set(y_all)) > 1 else float("nan")
    brier = brier_score_loss(y_all, p_all)
    ll = log_loss(y_all, [min(max(p, 1e-6), 1 - 1e-6) for p in p_all], labels=[0, 1])
    acc_ci, auc_ci = _bootstrap_ci(y_all, p_all)
    majority = max(sum(y_all), len(y_all) - sum(y_all)) / len(y_all)

    idx = 0
    for f in folds:
        seg = y_all[idx:idx + f["n_test"]]
        seg_p = p_all[idx:idx + f["n_test"]]
        f["acc"] = round(sum(1 for a, b in zip(seg, seg_p) if (b >= 0.5) == (a == 1)) / len(seg), 3)
        idx += f["n_test"]

    return WinEval(
        n=len(y_all), acc=acc, auc=auc, brier=brier, logloss=ll,
        acc_ci=acc_ci, auc_ci=auc_ci, baseline_acc=majority, folds=folds,
        note="按系列赛留一（LOSO）；同一场 BO5 不会跨训练/测试集"
             + ("；含队伍强度列（赛前已知，与阵容无关）" if use_team else ""),
    )
