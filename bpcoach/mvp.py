# -*- coding: utf-8 -*-
"""M3 的 MVP 支路：给定十手阵容，预测这一局谁最可能拿 MVP。

为什么值得单独做
    胜率告诉你「这套阵容该赢」，MVP 告诉你「这套阵容里谁会兑现它」。
    对教练来说这两件事的用法完全不同：前者决定要不要选，后者决定要围绕谁打。
    而且 MVP 数据本身给了模型一个**不依赖比分**的监督信号：
    同一个英雄在不同人手里拿 MVP 的概率差别很大（carry 位 vs 工具人），
    这正是「阵容价值落在谁身上」的度量。

一个必须承认的局限
    MVP 只发给胜方（1447 局里 1439 局有 MVP，与胜方一致率 100%），
    所以本模型天然偏向强队的选手。评估按系列赛留一，且对比基线是
    「永远猜最常拿 MVP 的那个选手」，不是「随机猜 10 个」——
    拿 10% 当基线是在自欺欺人。

做法
    候选级(pointwise)逻辑回归：把一局的 10 名选手各展开成一行，
    标签 1 = 本局 MVP，特征全部只用到「赛前可知」或「历史统计」：
        hero_mvp     该英雄历史上被选后拿 MVP 的概率（带收缩）
        player_mvp   该选手历史上拿 MVP 的比例（带收缩）
        hero_score   该英雄的平均 MVP 评分（选手评分，反映"容易刷分"）
        player_score 该选手的平均 MVP 评分
        lane_*       五条分路 one-hot（不同位置的评分分布不同）
    预测时对 10 个候选做 softmax，排序输出。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .dataset import Dataset, GameSample
from .heroes import HeroPool
from .series import BLUE, RED

# 收缩强度：一个英雄一季出场几十次，取 6 相当于「先验占 6 场」
SHRINK_HERO = 6.0
SHRINK_PLAYER = 6.0
# MVP 的基础率：10 个候选里出 1 个
BASE_RATE = 0.1


@dataclass
class MVPPriors:
    hero_mvp: list[float] = field(default_factory=list)
    hero_score: list[float] = field(default_factory=list)
    player_mvp: dict[str, float] = field(default_factory=dict)
    player_score: dict[str, float] = field(default_factory=dict)
    base_score: float = 8.0
    n_games: int = 0
    n_mvp: int = 0

    @classmethod
    def fit(cls, dataset: Dataset) -> "MVPPriors":
        pool = dataset.pool
        n = pool.size
        app = [0.0] * n
        mvp = [0.0] * n
        score_sum = [0.0] * n
        score_n = [0.0] * n
        p_app: dict[str, float] = {}
        p_mvp: dict[str, float] = {}
        p_score: dict[str, float] = {}
        p_score_n: dict[str, float] = {}
        games = mvp_games = 0
        all_scores: list[float] = []

        for _, g in dataset.labeled_games():
            if not g.players:
                continue
            games += 1
            for p in g.players:
                try:
                    hid = pool.index(p["hero"])
                except Exception:                       # noqa: BLE001
                    continue
                app[hid] += 1
                sc = float(p.get("mvp_score") or 0.0)
                if sc > 0:
                    score_sum[hid] += sc
                    score_n[hid] += 1
                    all_scores.append(sc)
                if p.get("is_mvp"):
                    mvp[hid] += 1
                    mvp_games += 1
                name = p.get("player") or ""
                if name:
                    p_app[name] = p_app.get(name, 0.0) + 1
                    p_mvp[name] = p_mvp.get(name, 0.0) + (1.0 if p.get("is_mvp") else 0.0)
                    if sc > 0:
                        p_score[name] = p_score.get(name, 0.0) + sc
                        p_score_n[name] = p_score_n.get(name, 0.0) + 1

        base_score = sum(all_scores) / len(all_scores) if all_scores else 8.0
        return cls(
            hero_mvp=[(mvp[i] + SHRINK_HERO * BASE_RATE) / (app[i] + SHRINK_HERO)
                      for i in range(n)],
            hero_score=[(score_sum[i] + 3 * base_score) / (score_n[i] + 3) for i in range(n)],
            player_mvp={k: (p_mvp.get(k, 0.0) + SHRINK_PLAYER * BASE_RATE)
                        / (v + SHRINK_PLAYER) for k, v in p_app.items()},
            player_score={k: (p_score.get(k, 0.0) + 3 * base_score)
                          / (p_score_n.get(k, 0.0) + 3) for k in p_app},
            base_score=base_score,
            n_games=games,
            n_mvp=mvp_games,
        )


LANES = ("对抗路", "打野", "中路", "发育路", "游走")


def candidate_features(pool: HeroPool, priors: MVPPriors, p: dict) -> tuple[list[float], list[str]]:
    """一名候选的特征。**只允许赛前可知或历史统计**，不能碰 k/d/a 与 gold。"""
    hid = pool.index(p["hero"])
    lm = pool.lane_model()
    probs = lm.probs(hid)
    top = max(range(len(probs)), key=lambda i: probs[i])
    name = p.get("player") or ""
    row = [
        priors.hero_mvp[hid],
        priors.player_mvp.get(name, BASE_RATE),
        (priors.hero_score[hid] - priors.base_score),
        (priors.player_score.get(name, priors.base_score) - priors.base_score),
    ] + [1.0 if i == top else 0.0 for i in range(len(lm.lanes))]
    names = ["hero_mvp", "player_mvp", "hero_score", "player_score"] + \
            [f"lane_{l}" for l in lm.lanes]
    return row, names


@dataclass
class MVPPrediction:
    ranking: list[tuple[str, str, float]]     # (选手, 英雄, 概率)，按概率降序
    top1: str
    feature_names: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.ranking:
            return "MVP 预测：无候选"
        name, hero, p = self.ranking[0]
        rest = "、".join(f"{n}({h}){q:.0%}" for n, h, q in self.ranking[1:3])
        return f"MVP 最可能：{name} 的 {hero}（{p:.0%}）" + (f"；其后 {rest}" if rest else "")


class MVPModel:
    """条件 logit（10 路 softmax）：同一局的 10 个候选互相竞争一个 MVP 名额。

    为什么不用普通的候选级逻辑回归
        逐个候选独立建模时，「1 个正例 + 9 个负例」的结构被丢掉了，
        预测出来再手工 softmax 归一化，概率会虚高（实测出现过 62% 这种数）。
        条件 logit 直接把 P(MVP = i | 这 10 个人) = exp(θ·x_i) / Σ exp(θ·x_j)
        写进似然，分母自带约束，概率天然在 10 个候选间归一。
    """

    def __init__(self, pool: HeroPool, seed: int = 42, l2: float = 1.0) -> None:
        self.pool = pool
        self.seed = seed
        self.l2 = l2
        self.priors: MVPPriors | None = None
        self.theta = None            # 权重（标准化后的特征空间）
        self.mean: list[float] = []
        self.std: list[float] = []
        self.feature_names: list[str] = []
        self.meta: dict = {}

    # ---------------- 训练 ----------------
    def _design(self, dataset: Dataset):
        """返回 (X, group_sizes, y_index, names)。每组是一局，y_index 是组内 MVP 的下标。"""
        X: list[list[float]] = []
        sizes: list[int] = []
        yidx: list[int] = []
        names = self.feature_names
        for _, g in dataset.labeled_games():
            if not g.players:
                continue
            rows, ok = [], []
            for p in g.players:
                try:
                    row, nm = candidate_features(self.pool, self.priors, p)
                except Exception:                       # noqa: BLE001
                    continue
                rows.append(row)
                ok.append(bool(p.get("is_mvp")))
                names = nm
            if len(rows) < 2 or not any(ok):
                continue
            X += rows
            sizes.append(len(rows))
            yidx.append(ok.index(True))
        self.feature_names = names
        return X, sizes, yidx

    def fit(self, dataset: Dataset) -> "MVPModel":
        import numpy as np
        from scipy.optimize import minimize

        self.priors = MVPPriors.fit(dataset)
        X, sizes, yidx = self._design(dataset)
        if not X:
            raise ValueError("没有 MVP 标签，无法训练 MVP 模型")
        arr = np.asarray(X, dtype=float)
        self.mean = arr.mean(axis=0).tolist()
        std = arr.std(axis=0)
        std[std < 1e-9] = 1.0
        self.std = std.tolist()
        Z = (arr - np.asarray(self.mean)) / np.asarray(self.std)

        # 组内切片，供目标函数按局做 softmax
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        flat_y = (starts + np.asarray(yidx)).tolist()
        d = Z.shape[1]

        def nll(theta: np.ndarray) -> float:
            tot = 0.0
            for s, n, k in zip(starts, sizes, yidx):
                seg = Z[s:s + n] @ theta
                seg = seg - seg.max()
                tot -= seg[k] - np.log(np.exp(seg).sum())
            return tot / len(sizes) + self.l2 * float(theta @ theta) / len(sizes)

        def grad(theta: np.ndarray) -> np.ndarray:
            g = np.zeros(d)
            for s, n, k in zip(starts, sizes, yidx):
                seg = Z[s:s + n] @ theta
                seg = np.exp(seg - seg.max())
                p = seg / seg.sum()
                p[k] -= 1.0
                g += p @ Z[s:s + n]
            return g / len(sizes) + 2 * self.l2 * theta / len(sizes)

        res = minimize(nll, np.zeros(d), jac=grad, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.theta = res.x
        self.meta = {"n_candidates": len(X), "n_games": len(sizes),
                     "ll_per_game": round(float(res.fun), 4),
                     "base_score": round(self.priors.base_score, 3),
                     "converged": bool(res.success)}
        return self

    # ---------------- 预测 ----------------
    def _scores(self, players: list[dict], priors: MVPPriors) -> tuple[list[float], list[tuple[str, str]]]:
        import numpy as np

        rows, keys = [], []
        for p in players:
            try:
                row, _ = candidate_features(self.pool, priors, p)
            except Exception:                           # noqa: BLE001
                continue
            rows.append(row)
            keys.append((p.get("player") or "", p["hero"]))
        if not rows:
            return [], keys
        Z = (np.asarray(rows, dtype=float) - np.asarray(self.mean)) / np.asarray(self.std)
        return [float(z) for z in Z @ self.theta], keys

    def predict(self, players: list[dict], priors: MVPPriors | None = None) -> MVPPrediction:
        import numpy as np

        pr = priors or self.priors
        logits, keys = self._scores(players, pr)
        if not logits:
            return MVPPrediction([], "", self.feature_names)
        z = np.asarray(logits)
        p = np.exp(z - z.max())
        p = p / p.sum()
        ranked = sorted(zip(keys, p.tolist()), key=lambda kv: -kv[1])
        ranking = [(n, h, q) for (n, h), q in ranked]
        return MVPPrediction(ranking, ranking[0][0] if ranking else "", self.feature_names)

    def predict_game(self, game: GameSample) -> MVPPrediction:
        return self.predict(list(game.players))


# ====================== 评估 ======================
def loso_mvp_evaluate(dataset: Dataset, max_groups: int | None = None) -> dict:
    """按系列赛留一评估 MVP 预测。

    指标是 Top-1 / Top-3 命中率（10 个候选里挑），基线有两条：
        uniform        10%（真的随机）
        top_player     永远猜「历史上最常拿 MVP 的选手」（这条才有意义）
    """
    from .dataset import Dataset as DS

    rows = [g for _, g in dataset.labeled_games() if g.players]
    groups = sorted({g.series_id for g in rows})
    if max_groups and len(groups) > max_groups:
        step = len(groups) / max_groups
        groups = [groups[int(i * step)] for i in range(max_groups)]

    hit1 = hit3 = total = 0
    base1 = 0
    by_lane: dict[str, list[int]] = {}
    for hold in groups:
        train = DS(dataset.pool, [s for s in dataset.series if s.id != hold], dataset.meta)
        try:
            m = MVPModel(dataset.pool).fit(train)
        except ValueError:
            continue
        best_name = (max(m.priors.player_mvp, key=m.priors.player_mvp.get)
                     if m.priors and m.priors.player_mvp else "")
        for g in rows:
            if g.series_id != hold:
                continue
            pred = m.predict_game(g)
            if not pred.ranking:
                continue
            truth = next((p for p in g.players if p.get("is_mvp")), None)
            if truth is None:
                continue
            total += 1
            tk = (truth.get("player") or "", truth["hero"])
            hit1 += pred.ranking[0][:2] == tk
            hit3 += tk in [r[:2] for r in pred.ranking[:3]]
            # 基线：永远猜训练折里最常拿 MVP 的那个选手（同队才算猜得中）
            ids = [(p.get("player") or "", p["hero"]) for p in g.players]
            base1 += best_name in [i[0] for i in ids] and ids[[i[0] for i in ids].index(
                best_name)] == tk if best_name else 0
            # 分层统计：真实 MVP 落在哪条路上，模型命中率如何
            lane = dataset.pool.lane_model().top_lanes(
                dataset.pool.index(truth["hero"]), 1)[0][0]
            rec = by_lane.setdefault(lane, [0, 0])
            rec[0] += 1
            rec[1] += pred.ranking[0][:2] == tk

    return {
        "n_games": total,
        "top1": round(hit1 / total, 4) if total else None,
        "top3": round(hit3 / total, 4) if total else None,
        "baseline_top1_top_player": round(base1 / total, 4) if total else None,
        "baseline_uniform": 0.1,
        "top1_by_true_lane": {k: {"n": v[0], "top1": round(v[1] / v[0], 3)}
                              for k, v in sorted(by_lane.items())},
        "note": "10 个候选里挑 1 个；top_player 基线 = 永远猜训练折里最常拿 MVP 的选手",
    }


# ====================== 持久化 ======================
def save(model: MVPModel, path: str | pathlib.Path) -> None:
    import joblib

    joblib.dump({"theta": model.theta, "mean": model.mean, "std": model.std,
                 "priors": model.priors, "feature_names": model.feature_names,
                 "meta": model.meta, "l2": model.l2,
                 "pool_size": model.pool.size}, path)
    pathlib.Path(str(path)).with_suffix(".json").write_text(
        json.dumps({"feature_names": model.feature_names, "meta": model.meta},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: str | pathlib.Path, pool: HeroPool | None = None) -> MVPModel:
    import joblib

    blob = joblib.load(path)
    pool = pool or HeroPool.load()
    if blob["pool_size"] != pool.size:
        raise ValueError("英雄池大小不一致，id 会错位")
    m = MVPModel(pool, l2=blob.get("l2", 1.0))
    m.theta = blob["theta"]
    m.mean = blob["mean"]
    m.std = blob["std"]
    m.priors = blob["priors"]
    m.feature_names = blob["feature_names"]
    m.meta = blob["meta"]
    return m
