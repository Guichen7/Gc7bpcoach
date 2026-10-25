# -*- coding: utf-8 -*-
"""数据集层：把 B/P 数据集文件读成模型能用的对象，并负责「回放成局面」。

为什么单独一层
    M2（马尔可夫链）、M3（胜率）、M4（MCTS）都要读同一份赛事数据，但需要的东西不同：
    M2 要「某方在某个局面下选了谁」，M3 要「这十手阵容的胜负」，M4 要「可搜索的局面快照」。
    如果每个模块各自解析 JSON，字段口径迟早会分叉。所以这里统一：
      · SeriesSample / GameSample  是只读的数据视图
      · replay()                    把「集合形式」的对局还原成逐步 DraftState

两代 schema，靠 order_known 区分
    schema 2（官方接口，默认，data/kpl_official_2026.json）
        每局都有 bp_list 的 20 个动作，是**真实手顺序**，还带胜负、MVP、选手面板。
        replay() 按记录顺序逐手执行，返回 replay_order="ordered"。
        这是 M2/M3 指标可信的前提。
    schema 1（第三方战报，data/kpl_2026_finals.json，遗留兜底）
        战报只公布每方的「禁用集合」和「按分路排好的选用集合」，没有手顺序。
        回放时按「同一方、同一动作类型」的桶依次填格（canonical 回放）：
            蓝方 ban 5 个 → 依次进 5 个 ban 位；蓝方 pick 5 个 → 依次进 5 个 pick 位
        动作的（方、类型）绝对正确，同类内部的先后是重建的，
        所以 replay_order="canonical"，基于它的评估必须标明「顺序是重建的」。

两种数据都不完整时，宁可丢局并计数（meta["dropped_games"]），也不猜。
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Iterator

from .errors import UnknownHeroError
from .heroes import HeroPool, config_dir
from .series import BLUE, GameRecord, RED, SeriesContext
from .state import DraftState
from .template import BAN, PICK, BPTemplate


def default_dataset_path() -> pathlib.Path:
    """默认数据集：优先用官方逐局数据（有手顺序与 MVP），没有再退回第三方战报。

    官方文件由 tools/fetch_kpl_official.py 生成；两者 schema 不同但字段兼容，
    Dataset.load 都能读，所以这里只做「谁更像真数据就用谁」的排序。
    """
    root = config_dir().parents[0]
    official = root / "data" / "kpl_official_2026.json"
    if official.exists():
        return official
    return root / "data" / "kpl_2026_finals.json"


def default_results_path() -> pathlib.Path:
    """逐局胜负标签（由 tools/fetch_kpl_results.py 生成）。缺文件时 M3 无法训练，但 M1/M2 不受影响。"""
    return config_dir().parents[0] / "data" / "kpl_2026_results.json"


@dataclass(frozen=True)
class GameSample:
    """一局的 B/P 集合。hero 全部是英雄 id，与 HeroPool 对齐。"""

    series_id: str
    game_index: int
    blue: str
    red: str
    bans: dict[int, tuple[int, ...]]
    picks: dict[int, tuple[int, ...]]
    order_known: bool = False
    # 若数据源给了真实手顺序，这里形如 ((0, "ban", 1, 17), (1, "ban", 2, 5), ...)
    ordered: tuple[tuple[int, str, int, int], ...] = ()
    # 胜负标签（M3 的 y）："blue" / "red" / None。与 B/P 分开采集，缺失是常态。
    winner: str | None = None
    # 本局 MVP 英雄（若有）。既做数据校验（MVP 应属胜方），也做「谁有 carry 能力」的先验
    mvp_hero: int | None = None
    # 逐选手数据（官方接口才有）：hero/k/d/a/gold/hurt/mvp_score/is_mvp...
    # M3 用它做「阵容打完值多少」的辅助标签；预测时**不可**使用（赛后信息）
    players: tuple[dict, ...] = ()
    duration_s: int = 0

    @property
    def blue_side(self) -> int:
        return BLUE

    def heroes_of(self, side: int, kind: str) -> tuple[int, ...]:
        return self.bans[side] if kind == BAN else self.picks[side]

    def all_heroes(self) -> tuple[int, ...]:
        return (
            self.bans[BLUE] + self.bans[RED] + self.picks[BLUE] + self.picks[RED]
        )

    def to_game_record(self, index: int) -> GameRecord:
        return GameRecord(
            game_index=index,
            picks={BLUE: list(self.picks[BLUE]), RED: list(self.picks[RED])},
            bans={BLUE: list(self.bans[BLUE]), RED: list(self.bans[RED])},
        )


@dataclass
class SeriesSample:
    id: str
    date: str
    stage: str
    teams: list[str]
    games: list[GameSample] = field(default_factory=list)
    partial: bool = False


@dataclass
class ReplayStep:
    """回放出来的一手：用于评估时「看到第几步、真实选了什么」。"""

    step_index: int
    kind: str
    side: int
    hero_id: int
    state: DraftState          # 这一手**之前**的局面（可直接喂给模型）
    permissive: bool = False   # 该局因锁定冲突而按宽松模式回放


def _load_labels(path: str | pathlib.Path | None = None) -> dict[str, dict[int, dict]]:
    """读逐局标签，返回 {series_id: {game_index: {"winner":..., "mvp_raw":...}}}。

    标签是独立文件：B/P 与胜负来自不同站点、不同采集时间，
    合并成一个文件会让「哪些字段可信」失去追溯性（见 tools/fetch_kpl_results.py）。
    文件不存在时返回空字典 —— M1/M2 不需要标签，M3 会明确报「没有可用标签」。
    """
    p = pathlib.Path(path) if path else default_results_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict[int, dict]] = {}
    for s in raw.get("series", []):
        out[s["series_id"]] = {
            int(g["game_index"]): {"winner": g.get("winner"), "mvp_raw": g.get("mvp_raw")}
            for g in s.get("games", [])
        }
    return out


def _resolve_hero(pool: HeroPool, raw: str | None) -> int | None:
    """把「WB暖阳(林恒)马超」这类 MVP 文本解析成英雄 id：去括号后做最长后缀匹配。"""
    if not raw:
        return None
    text = re.sub(r"[（(][^）)]*[）)]", "", str(raw)).strip()
    best = None
    for hero in pool.heroes:
        for key in (hero.name, *hero.aliases):
            if text.endswith(key) and (best is None or len(key) > best[0]):
                best = (len(key), hero.id)
    return best[1] if best else None


class Dataset:
    def __init__(
        self,
        pool: HeroPool,
        series: list[SeriesSample],
        meta: dict | None = None,
    ) -> None:
        self.pool = pool
        self.series = series
        self.meta = meta or {}

    # ---------------- 构造 ----------------
    @staticmethod
    def _parse_game(pool: HeroPool, series_id: str, g: dict, lab: dict) -> GameSample:
        """解析一局。兼容两种 schema：

        schema 1（第三方战报，data/kpl_2026_finals.json）
            picks/bans 是名字列表，没有手顺序，胜负在另一个文件里按 series_id+局号对齐
        schema 2（官方接口，data/kpl_official_2026.json）
            order 里带每一步的（方, 禁/选, 英雄），winner / mvp / players 就在本局内
        """
        gi = int(g["game_index"])
        picks = {BLUE: tuple(pool.index(n) for n in g["picks"]["blue"]),
                 RED: tuple(pool.index(n) for n in g["picks"]["red"])}
        bans = {BLUE: tuple(pool.index(n) for n in g["bans"]["blue"]),
                RED: tuple(pool.index(n) for n in g["bans"]["red"])}

        if "order" in g:                      # ---- schema 2
            ordered = tuple(
                (int(o["step"]), o["kind"], BLUE if o["side"] == "blue" else RED,
                 pool.index(o["hero"]))
                for o in g["order"]
            )
            mvp = g.get("mvp") or {}
            return GameSample(
                series_id=series_id, game_index=gi, blue=g["blue"], red=g["red"],
                bans=bans, picks=picks,
                order_known=bool(ordered), ordered=ordered,
                winner=g.get("winner"),
                mvp_hero=pool.index(mvp["hero"]) if mvp.get("hero") else None,
                players=tuple(g.get("players") or ()),
                duration_s=int(g.get("duration_s") or 0),
            )

        # ---- schema 1
        return GameSample(
            series_id=series_id, game_index=gi, blue=g["blue"], red=g["red"],
            bans=bans, picks=picks,
            order_known=bool(g.get("order_known", False)),
            winner=lab.get("winner"),
            mvp_hero=_resolve_hero(pool, lab.get("mvp_raw")),
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path | None = None,
        pool: HeroPool | None = None,
        results_path: str | pathlib.Path | None = None,
    ) -> "Dataset":
        p = pathlib.Path(path) if path else default_dataset_path()
        raw = json.loads(p.read_text(encoding="utf-8"))
        pool = pool or HeroPool.load()

        labels = _load_labels(results_path)
        series: list[SeriesSample] = []
        unknown: dict[str, int] = {}
        dropped: list[dict] = []
        for s in raw["series"]:
            lab = labels.get(s["id"], {})
            games: list[GameSample] = []
            for g in s["games"]:
                try:
                    games.append(cls._parse_game(pool, s["id"], g, lab.get(int(g["game_index"]), {})))
                except UnknownHeroError as exc:
                    # 英雄池缺这个英雄时不能静默塞个 0 号位进去——那会把别的英雄的
                    # 统计悄悄混进来。宁可丢掉这一局，把缺失名字暴露到 meta 里。
                    key = getattr(exc, "raw", str(exc)) if hasattr(exc, "raw") else str(exc)
                    unknown[str(key)] = unknown.get(str(key), 0) + 1
                    dropped.append({"series": s["id"], "game": int(g["game_index"]),
                                    "reason": str(exc)})
            if not games:
                continue
            series.append(
                SeriesSample(
                    id=s["id"],
                    date=s.get("date", ""),
                    stage=s.get("stage", ""),
                    teams=list(s.get("teams", [])),
                    games=games,
                    partial=bool(s.get("partial", False)),
                )
            )
        meta = {k: v for k, v in raw.items() if k != "series"}
        meta["path"] = str(p)
        meta["unknown_heroes"] = unknown
        meta["dropped_games"] = {k: v for k, v in
                                 (("count", len(dropped)), ("detail", dropped[:20]))}
        meta["labels"] = {
            "path": str(results_path or default_results_path()),
            "series_with_labels": len(labels),
        }
        return cls(pool, series, meta)

    # ---------------- 查询 ----------------
    @property
    def template_id(self) -> str:
        return self.meta.get("template", "kpl_2026_bo5_global")

    @property
    def order_known(self) -> bool:
        return bool(self.meta.get("order_known", False))

    def games(self) -> Iterator[GameSample]:
        for s in self.series:
            yield from s.games

    @property
    def n_games(self) -> int:
        return sum(len(s.games) for s in self.series)

    def stats(self) -> dict:
        per_kind = {BAN: 0, PICK: 0}
        heroes: dict[int, int] = {}
        for g in self.games():
            per_kind[BAN] += len(g.bans[BLUE]) + len(g.bans[RED])
            per_kind[PICK] += len(g.picks[BLUE]) + len(g.picks[RED])
            for h in g.all_heroes():
                heroes[h] = heroes.get(h, 0) + 1
        return {
            "series": len(self.series),
            "games": self.n_games,
            "slots": per_kind,
            "heroes_used": len(heroes),
            "order_known": self.order_known,
        }

    def top_heroes(self, kind: str | None = None, limit: int = 10) -> list[tuple[str, int]]:
        """出场/禁用频次榜，用于人工核对数据是否合理。"""
        cnt: dict[int, int] = {}
        for g in self.games():
            for side in (BLUE, RED):
                for k in ({kind} if kind else {BAN, PICK}):
                    for h in g.heroes_of(side, k):
                        cnt[h] = cnt.get(h, 0) + 1
        ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [(self.pool.name_of(h), c) for h, c in ranked]

    # ---------------- 标签 ----------------
    def labeled_games(self) -> list[tuple[SeriesSample, GameSample]]:
        """只保留有胜负标签的对局。M3 的训练与评估都从这里取。"""
        return [
            (s, g) for s in self.series for g in s.games if g.winner in ("blue", "red")
        ]

    @property
    def n_labeled(self) -> int:
        return len(self.labeled_games())

    def label_stats(self) -> dict:
        """标签覆盖率与自洽性。

        MVP 校验值得单列：MVP 按惯例属于胜方（联盟颁奖给赢的那一边），
        所以「MVP 英雄出现在负方」是**数据打架的强信号**——
        要么 B/P 对错了场次，要么比分差分推错了赢家。它的错误率必须被看见。
        """
        labeled = self.labeled_games()
        blue_wins = sum(1 for _, g in labeled if g.winner == "blue")
        mvp_total = mvp_ok = 0
        bad: list[dict] = []
        for s, g in labeled:
            if g.mvp_hero is None:
                continue
            mvp_total += 1
            side = BLUE if g.mvp_hero in g.picks[BLUE] else (RED if g.mvp_hero in g.picks[RED] else None)
            if side is None:
                continue
            if (side == BLUE) == (g.winner == "blue"):
                mvp_ok += 1
            else:
                bad.append({
                    "series": s.id, "game": g.game_index,
                    "mvp": self.pool.name_of(g.mvp_hero), "winner": g.winner,
                })
        return {
            "games": self.n_games,
            "labeled": len(labeled),
            "coverage": round(len(labeled) / self.n_games, 4) if self.n_games else 0.0,
            "blue_win_rate": round(blue_wins / len(labeled), 4) if labeled else None,
            "series": len({s.id for s, _ in labeled}),
            "mvp_available": mvp_total,
            "mvp_agrees_with_winner": mvp_ok,
            "mvp_conflicts": bad[:10],
        }

    # ---------------- 回放 ----------------
    def slot_plan(
        self, game: GameSample, template: BPTemplate
    ) -> list[tuple[int, str, int, int]]:
        """把一局的对局集合铺到模板的 20 个格子上。

        order_known=true  -> 直接用记录里的真实顺序
        order_known=false -> canonical：同方同类按发布顺序依次入格
        """
        if game.order_known and len(game.ordered) == template.total_steps:
            return list(game.ordered)

        buckets: dict[tuple[int, str], list[int]] = {}
        for side in (BLUE, RED):
            buckets[(side, BAN)] = list(game.bans[side])
            buckets[(side, PICK)] = list(game.picks[side])

        plan: list[tuple[int, str, int, int]] = []
        for step in template.steps:
            queue = buckets.get((step.side, step.kind))
            if not queue:
                raise ValueError(
                    f"{game.series_id} 第 {game.game_index} 局缺少 "
                    f"{'蓝' if step.side == BLUE else '红'}方 {step.kind} 位"
                )
            plan.append((step.index, step.kind, step.side, queue.pop(0)))
        return plan

    def series_context(self, series: SeriesSample, before_game_index: int) -> SeriesContext:
        """用**本系列赛里已有**的、局号更小的对局构建全局 B/P 上下文。

        数据可能缺局（例如只抓到第 2~5 局），缺局时上下文只能少算，不会算错：
        漏掉的锁定只会让「可用集合」偏大，不会把合法动作判成非法。
        """
        ctx = SeriesContext(self.pool, global_lock=self.global_lock)
        if self.global_lock == "none":
            return ctx
        usable = [g for g in series.games if g.game_index < before_game_index]
        for i, g in enumerate(usable, start=1):
            ctx.commit(g.to_game_record(i))
        return ctx

    # ---------------- 数据自洽性检查 ----------------
    @property
    def global_lock(self) -> str:
        return self.meta.get("global_lock", "picks")

    def lock_conflicts(self) -> list[dict]:
        """检查数据是否符合「每个英雄每一方只能使用一次」。

        这是拿规则去检验数据的硬约束：同一方在后续局里重复选到前几局用过的英雄，
        说明要么数据源不准，要么该赛段其实没有启用全局 B/P。两种都要让人看见，
        不能悄悄放过——否则训练的转移矩阵会建立在互相矛盾的样本上。
        """
        out: list[dict] = []
        for s in self.series:
            used = {BLUE: set(), RED: set()}
            for g in s.games:
                for side in (BLUE, RED):
                    for h in g.picks[side]:
                        if h in used[side]:
                            out.append({
                                "series": s.id,
                                "game": g.game_index,
                                "side": "blue" if side == BLUE else "red",
                                "hero": self.pool.name_of(h),
                                "type": "same_side_pick_reuse",
                            })
                for side in (BLUE, RED):
                    used[side] |= set(g.picks[side])
        return out

    def violates_lock(self, series: SeriesSample, game: GameSample) -> bool:
        if self.global_lock == "none":
            return False
        ctx = self.series_context(series, game.game_index)
        return any(ctx.lock_of(h, side) for side in (BLUE, RED) for h in game.picks[side])

    def replay(
        self,
        game: GameSample,
        template: BPTemplate,
        series: SeriesSample | None = None,
        our_side: int = BLUE,
    ) -> list[ReplayStep]:
        """逐步回放一局，每一步都给出「该手之前的局面」。

        若该局与「前几局锁定」冲突（见 lock_conflicts），回放会自动降级为
        「不带跨局上下文」的宽松模式：这样能继续产出可用的转移样本，
        同时冲突本身由 lock_conflicts() 单独暴露，不会被静默吞掉。
        """
        ctx = None
        permissive = False
        if series is not None and self.global_lock != "none":
            if self.violates_lock(series, game):
                permissive = True
            else:
                ctx = self.series_context(series, game.game_index)
        state = DraftState(self.pool, template, our_side=our_side, series=ctx)
        steps: list[ReplayStep] = []
        for step_index, kind, side, hero in self.slot_plan(game, template):
            steps.append(ReplayStep(step_index, kind, side, hero, state.clone()))
            state.apply(hero, kind=kind)
        if steps:
            steps[0].permissive = permissive
        return steps

    def iter_replays(self, template: BPTemplate) -> Iterator[tuple[SeriesSample, GameSample, list[ReplayStep]]]:
        for s in self.series:
            for g in s.games:
                yield s, g, self.replay(g, template, series=s)

    def __repr__(self) -> str:
        return (
            f"<Dataset {self.meta.get('dataset')} series={len(self.series)} "
            f"games={self.n_games} order_known={self.order_known}>"
        )
