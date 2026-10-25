# -*- coding: utf-8 -*-
"""M1 局面编码：把一局 B/P 变成模型和搜索都能直接吃的东西。

一、编码约定的由来
论文里 agent 只能用「110 维状态数组」：被 ban 记 -1，被我方选走记 1，被对方选走记 2。
这个约定的好处是**与先手侧无关**——不管我们是蓝方还是红方，模型看到的永远是
「我方 1 / 敌方 2」，先手优势另行作为标量特征给进去，不会和英雄身份混在一列里。
所以本模块内部按蓝/红存储（贴合真实规则），对外统一换算成我方/敌方。

二、为什么要三套向量
    encode()          单局棋盘，带符号（-1/0/1/2）——给可视化、给状态键、给人看
    encode_pair()     我方 0/1 + 敌方 0/1，各 N 维 —— 论文 M3 的输入就是这两段拼接
    available_mask()  还能不能选 —— M4 展开子节点时先乘这个掩码，避免生成非法分支
    encode_used()     本系列赛已消耗 —— M2/M3 判断全局 B/P 的关键特征

三、状态键
    state_key() 只包含「走到第几步 + 当前棋盘」，不含历史动作顺序，
    因为 B/P 的先后手由模板唯一决定，同一棋盘不可能有不同的后续。这让 M4 的
    置换表能真正复用节点。
"""

from __future__ import annotations

from .errors import BPCoachError, DraftFinishedError, IllegalActionError
from .heroes import HeroPool
from .series import GameRecord, SeriesContext, other
from .template import BAN, PICK, BPTemplate, Step

# 棋盘取值（注意与 template.BAN / template.PICK 这两个「动作类型」字符串区分开）
EMPTY = 0
BANNED = -1
PICKED = 1        # encode() 里 1 恒表示「我方」
OPP_PICKED = 2    # encode() 里 2 恒表示「敌方」

# 内部按蓝红存储（与 series.BLUE / series.RED 同值，这里保留便于 type 注解阅读）
BLUE = 1
RED = 2


class DraftState:
    """一局 B/P 的不可变语义（操作可变，但每次 apply 只前进一格，可 undo）。"""

    __slots__ = (
        "pool", "template", "our_side", "series", "game_index", "patch",
        "_board", "_cursor", "_history",
    )

    def __init__(
        self,
        pool: HeroPool,
        template: BPTemplate,
        our_side: int = BLUE,
        series: SeriesContext | None = None,
        game_index: int | None = None,
        patch: str | None = None,
    ) -> None:
        if our_side not in (BLUE, RED):
            raise BPCoachError("our_side 只能是 1（蓝方）或 2（红方）")
        self.pool = pool
        self.template = template
        self.our_side = our_side
        self.series = series
        self.game_index = game_index if game_index is not None else (
            series.game_index if series else 1
        )
        self.patch = patch if patch is not None else pool.patch

        self._board: list[int] = [EMPTY] * pool.size
        self._cursor = 0
        self._history: list[tuple[int, int, str, int]] = []

    # ================= 局面只读视图 =================
    @property
    def step_index(self) -> int:
        """已经执行了多少手（= 下一步的下标）。"""
        return self._cursor

    @property
    def total_steps(self) -> int:
        return self.template.total_steps

    @property
    def is_finished(self) -> bool:
        return self._cursor >= self.template.total_steps

    @property
    def current_step(self) -> Step | None:
        return None if self.is_finished else self.template.steps[self._cursor]

    @property
    def side_to_act(self) -> int | None:
        step = self.current_step
        return step.side if step else None

    @property
    def kind_to_act(self) -> str | None:
        step = self.current_step
        return step.kind if step else None

    @property
    def is_our_turn(self) -> bool:
        return self.side_to_act == self.our_side

    @property
    def last_action(self) -> tuple[int, str, int] | None:
        """上一手：(英雄 id, ban/pick, 行动方)。没有动作时返回 None。"""
        if not self._history:
            return None
        _, hid, kind, side = self._history[-1]
        return hid, kind, side

    def picks_of(self, side: int) -> list[int]:
        return [i for i, v in enumerate(self._board) if v == side]

    def bans(self) -> list[int]:
        return [i for i, v in enumerate(self._board) if v == BANNED]

    def _picked_ids(self) -> set[int]:
        return {i for i, v in enumerate(self._board) if v in (BLUE, RED)}

    def pick_group_block(self, hero_id: int) -> int | None:
        """选这张卡是否因「同组已选」而不能选；返回撞上的那张卡的 id。

        元流之子的 5 张职业卡共用 1 个选用名额，但 ban 是分开的：
        ban 掉「辅助」不影响选「坦克」。所以这条只在 PICK 时检查。
        """
        return self.pool.pick_excludes(hero_id, self._picked_ids())

    def board(self) -> list[int]:
        """原始蓝红棋盘（1 蓝 / 2 红 / -1 ban / 0 空）。"""
        return list(self._board)

    def clone(self) -> "DraftState":
        new = DraftState(
            self.pool, self.template, self.our_side,
            self.series.clone() if self.series else None,
            self.game_index, self.patch,
        )
        new._board = list(self._board)
        new._cursor = self._cursor
        new._history = list(self._history)
        return new

    # ================= 合法性 =================
    def _locked_for(self, side: int, hero_id: int) -> bool:
        """该英雄是否已被 side **自己**用掉（全局 B/P）。"""
        return bool(self.series and self.series.lock_of(hero_id, side))

    def is_wasted_ban(self, hero_id: int, side: int | None = None) -> bool:
        """ban 一个「对方已经用过」的英雄 —— 合法但等于空过。"""
        if self.series is None:
            return False
        banner = side if side is not None else self.side_to_act
        if banner is None:
            return False
        return self.series.ban_is_wasted(hero_id, banner)

    def is_legal(self, hero: str | int, exclude_wasted: bool = True) -> bool:
        hid = self.pool.index(hero)
        if self.is_finished:
            return False
        if self._board[hid] != EMPTY:
            return False
        step = self.current_step
        if step.kind == PICK and self.pick_group_block(hid) is not None:
            return False
        if self._locked_for(step.side, hid):
            # 选人：本方用过，规则禁止；ban：只有对方用过才没意义，分情况处理
            if step.kind == PICK:
                return False
            return not exclude_wasted and not self.is_wasted_ban(hid, step.side)
        return True

    def legal_hero_ids(self, exclude_wasted: bool = True) -> list[int]:
        """当前这一步真正值得考虑的动作集合。

        exclude_wasted=True 时剔除「对方已用过、ban 了也没意义」的英雄。
        搜推（M4）应该用 True；做规则校验/复盘回放时用 False。
        """
        if self.is_finished:
            return []
        step = self.current_step
        picked = self._picked_ids() if step.kind == PICK else set()
        out: list[int] = []
        for hid, v in enumerate(self._board):
            if v != EMPTY:
                continue
            if picked and self.pool.pick_excludes(hid, picked) is not None:
                continue
            if self._locked_for(step.side, hid):
                if step.kind == PICK:
                    continue
                if exclude_wasted:
                    continue
            out.append(hid)
        return out

    def legal_names(self) -> list[str]:
        return self.pool.names(self.legal_hero_ids())

    # ================= 执行 =================
    def apply(self, hero: str | int, kind: str | None = None) -> Step:
        """按其所在步执行一手（禁或选由模板决定，不用自己判断）。"""
        if self.is_finished:
            raise DraftFinishedError("B/P 已经走完，无法继续执行动作")

        step = self.template.steps[self._cursor]
        if kind is not None and kind != step.kind:
            raise IllegalActionError(
                f"第 {self._cursor + 1} 手应为 {step.kind}，收到 {kind}"
            )
        hid = self.pool.index(hero)

        if self._board[hid] != EMPTY:
            raise IllegalActionError(f"{self.pool.name_of(hid)} 在本局已被 ban 或被选")
        if step.kind == PICK and self._locked_for(step.side, hid):
            raise IllegalActionError(
                f"{self.pool.name_of(hid)} 已被本方在前几局使用，按全局 B/P 规则不能再选"
            )
        if step.kind == PICK:
            blocker = self.pick_group_block(hid)
            if blocker is not None:
                raise IllegalActionError(
                    f"{self.pool.name_of(hid)} 与场上已选的 "
                    f"{self.pool.name_of(blocker)} 共用同一个选用名额，不能同时选"
                )
        # 注意：ban 一个「对方已用过」的英雄是**合法但空过**的动作。
        # 规则 6.2(A) 说不该出现在对方禁用池里，但游戏内并不阻止点它，
        # 真实赛事数据里也确实出现（见 data_quality 说明）。
        # 因此这里不报错，只用 is_wasted_ban() 标记，是否从候选集里剔除交给上层
        # （M4 搜索默认剔除以省算力，回放/统计必须保留）。

        self._board[hid] = BANNED if step.kind == BAN else step.side
        self._cursor += 1
        self._history.append((step.index, hid, step.kind, step.side))
        return step

    def apply_many(self, heroes) -> list[Step]:
        return [self.apply(h) for h in heroes]

    def undo(self) -> tuple[int, int, str, int] | None:
        if not self._history:
            return None
        step_index, hid, kind, side = self._history.pop()
        self._board[hid] = EMPTY
        self._cursor = step_index
        return step_index, hid, kind, side

    # ================= 编码：核心 =================
    def encode(self) -> list[int]:
        """单局棋盘，长度 N。0 未涉及 / -1 被 ban / 1 我方 / 2 敌方。"""
        if self.our_side == BLUE:
            return list(self._board)
        flip = {EMPTY: EMPTY, BANNED: BANNED, BLUE: RED, RED: BLUE}
        return [flip[v] for v in self._board]

    def encode_pair(self) -> tuple[list[int], list[int]]:
        """我方 / 敌方两个 0/1 向量，各 N 维 —— M3 胜率模型的标准输入。"""
        ours = [0] * self.pool.size
        opp = [0] * self.pool.size
        for hid, v in enumerate(self.encode()):
            if v == PICKED:
                ours[hid] = 1
            elif v == OPP_PICKED:
                opp[hid] = 1
        return ours, opp

    def encode_banned(self) -> list[int]:
        return [1 if v == BANNED else 0 for v in self._board]

    def encode_used(self) -> list[int]:
        """我方在本系列赛已消耗的英雄，长度 N。"""
        return self.series.used_vector(self.our_side) if self.series else [0] * self.pool.size

    def encode_used_opp(self) -> list[int]:
        """敌方已消耗的英雄，长度 N。

        为什么必须和「我方已消耗」分开：2026 规则的锁是按队伍算的，
        对方用过的英雄我方**仍然可以选**。合成一条会把这个信息抹掉。
        """
        if self.series is None:
            return [0] * self.pool.size
        return self.series.used_vector(other(self.our_side))

    def available_mask(self, side: int | None = None) -> list[int]:
        """还能被 `side` 选走的英雄，长度 N。M4 展开分支前先乘它。

        side 省略时取当前行动方；已结束时退化为我方。
        """
        who = side if side is not None else (self.side_to_act or self.our_side)
        used = self.series.used_vector(who) if self.series else [0] * self.pool.size
        # 当前这一手是「选」时，同组已选的卡也不能再选（元流之子的 5 张职业卡）
        picked = self._picked_ids()
        is_pick = (not self.is_finished) and self.current_step.kind == PICK
        out = []
        for i, v in enumerate(self._board):
            if v != EMPTY or used[i]:
                out.append(0)
                continue
            if is_pick and self.pool.pick_excludes(i, picked) is not None:
                out.append(0)
                continue
            out.append(1)
        return out

    def lane_profile(self) -> dict[str, list[float]]:
        """双方各分路的**期望占用**，用于判断阵容是否畸形（不是硬约束，是特征）。

        摇摆位在这里必须按概率摊开：马超「对抗 75% / 打野 25%」意味着
        选了他之后，两条路各被占去一部分概率，而不是两条路同时被占满。
        数学上这是把当前选人单射到分路的所有方案按概率加权（见 lane.py），
        所以 Σ_l occupancy_l 恒等于已选人数，不会因为摇摆位凭空多占位置。
        """
        lanes = list(self.pool.lanes)
        model = self.pool.lane_model()
        ours = model.occupancy(self.picks_of(self.our_side))
        opp = model.occupancy(self.picks_of(RED if self.our_side == BLUE else BLUE))
        return {"lanes": lanes, "ours": ours, "opp": opp}

    def meta(self) -> dict:
        """标量特征：先手侧、进度、系列赛比分、剩余手数等。"""
        step = self.current_step
        ours_picks = len(self.picks_of(self.our_side))
        opp_side = RED if self.our_side == BLUE else BLUE
        score = self.series.score() if self.series else {1: 0, 2: 0}
        return {
            "step_index": self._cursor,
            "step_total": self.total_steps,
            "progress": round(self._cursor / self.total_steps, 4) if self.total_steps else 0.0,
            "kind_to_act": step.kind if step else None,
            "side_to_act": step.side if step else None,
            "is_our_turn": self.is_our_turn,
            "our_side": "blue" if self.our_side == BLUE else "red",
            "our_picks": ours_picks,
            "opp_picks": len(self.picks_of(opp_side)),
            "total_bans": len(self.bans()),
            "remaining_our_picks": self.template.count(self.our_side, PICK) - ours_picks,
            "remaining_opp_picks": self.template.count(opp_side, PICK) - len(self.picks_of(opp_side)),
            "game_index": self.game_index,
            "games_total": self.template.games,
            "wins_needed": self.template.wins_needed,
            "series_score_ours": score.get(self.our_side, 0),
            "series_score_opp": score.get(opp_side, 0),
            "locked_ours": sum(self.encode_used()),
            "locked_opp": sum(self.encode_used_opp()),
            "patch": self.patch,
        }

    def model_input(
        self,
        *,
        include_banned: bool = False,
        include_used: bool = True,
        include_meta: bool = False,
        include_lanes: bool = False,
    ) -> dict:
        """打包给下游模型。

        默认「我方 0/1 + 敌方 0/1」两段拼接，与论文的 110+110 完全对应；
        其余块按需附加，避免模型输入维度随版本漂移。
        """
        ours, opp = self.encode_pair()
        flat = list(ours) + list(opp)
        names = [f"ours:{h.name}" for h in self.pool] + [f"opp:{h.name}" for h in self.pool]

        blocks = {"ours": len(ours), "opp": len(opp)}
        if include_banned:
            banned = self.encode_banned()
            flat += banned
            names += [f"ban:{h.name}" for h in self.pool]
            blocks["banned"] = len(banned)
        if include_used:
            used_ours = self.encode_used()
            used_opp = self.encode_used_opp()
            flat += used_ours + used_opp
            names += [f"used_ours:{h.name}" for h in self.pool]
            names += [f"used_opp:{h.name}" for h in self.pool]
            blocks["used_ours"] = len(used_ours)
            blocks["used_opp"] = len(used_opp)
        if include_lanes:
            prof = self.lane_profile()
            flat += prof["ours"] + prof["opp"]
            names += [f"ours_lane:{l}" for l in prof["lanes"]] + [
                f"opp_lane:{l}" for l in prof["lanes"]
            ]
            blocks["lanes"] = len(prof["ours"]) * 2
        if include_meta:
            m = self.meta()
            meta_vec = [
                m["progress"],
                1.0 if m["is_our_turn"] else 0.0,
                1.0 if m["our_side"] == "blue" else 0.0,
                m["remaining_our_picks"] / 5.0,
                m["remaining_opp_picks"] / 5.0,
                m["series_score_ours"] / max(m["wins_needed"], 1),
                m["series_score_opp"] / max(m["wins_needed"], 1),
            ]
            flat += meta_vec
            names += [
                "meta:progress", "meta:is_our_turn", "meta:is_blue",
                "meta:remaining_our_picks", "meta:remaining_opp_picks",
                "meta:score_ours", "meta:score_opp",
            ]
            blocks["meta"] = len(meta_vec)

        return {
            "x": flat,
            "feature_names": names,
            "blocks": blocks,
            "available": self.available_mask(),
            "meta": self.meta(),
            "dims": len(flat),
        }

    def state_key(self) -> tuple:
        """可哈希状态键：(步数, 我方/敌方棋盘)。供 M4 置换表去重。"""
        return (self._cursor, *self.encode())

    # ================= 校验与展示 =================
    def check(self) -> list[str]:
        """返回问题清单，空列表代表这一步局面合法。用于导入外部数据时兜底。"""
        issues: list[str] = []
        ours = len(self.picks_of(self.our_side))
        opp_side = RED if self.our_side == BLUE else BLUE
        opp = len(self.picks_of(opp_side))
        if ours > 5 or opp > 5:
            issues.append(f"选人数超限：我方 {ours}、敌方 {opp}（上限各 5）")

        expected_bans = sum(1 for s in self.template.steps[: self._cursor] if s.is_ban)
        if len(self.bans()) != expected_bans:
            issues.append(f"ban 数与进度不符：应有 {expected_bans}，实际 {len(self.bans())}")

        if self.series and self.template.global_lock != "none":
            # 1) 选用：本方自己用过的英雄不能再选（锁按队伍算，对方用过不影响）
            for side in (BLUE, RED):
                for hid in self.picks_of(side):
                    if self._locked_for(side, hid):
                        who = "我方" if side == self.our_side else "敌方"
                        issues.append(
                            f"{self.pool.name_of(hid)} 已被{who}在前几局使用，本局却仍被选用"
                            "（违反「每个英雄每一方只能使用一次」）"
                        )
            # 2) 禁用：ban 一个对方早已用过的英雄等于空过
            for _, hid, kind, side in self._history:
                if kind == BAN and self.is_wasted_ban(hid, side):
                    issues.append(
                        f"{self.pool.name_of(hid)} 是无效 ban：对方在前几局已用过该英雄"
                    )
        return issues

    def timeline(self) -> list[dict]:
        """按真实发生顺序返回每一手，带 b1/p2 这种赛事记法。"""
        out = []
        for step_index, hid, kind, side in self._history:
            step = self.template.steps[step_index]
            out.append({
                "step": step_index + 1,
                "label": step.order_in_phase,
                "kind": kind,
                "side": side,
                "hero_id": hid,
                "hero": self.pool.name_of(hid),
                "is_ours": side == self.our_side,
            })
        return out

    def board_text(self) -> str:
        """把当前棋盘摆成一条时间线，方便贴进日志或回给前端。"""
        ours_side = "蓝方" if self.our_side == BLUE else "红方"
        lines = [
            f"[第 {self.game_index} 局 / {self.template.id}] 我方={ours_side}  "
            f"进度 {self._cursor}/{self.total_steps}"
        ]
        tl = self.timeline()
        bans = [f"{a['label']} {a['hero']}" for a in tl if a["kind"] == BAN]
        lines.append("禁用：" + (" · ".join(bans) if bans else "—"))
        for side, label in ((BLUE, "蓝方"), (RED, "红方")):
            mark = "（我方）" if side == self.our_side else "（敌方）"
            picks = [f"{a['label']} {a['hero']}" for a in tl if a["kind"] == PICK and a["side"] == side]
            lines.append(f"{label}{mark}：" + (" · ".join(picks) if picks else "—"))
        if not self.is_finished:
            step = self.current_step
            who = "我方" if step.side == self.our_side else "敌方"
            side_name = "蓝方" if step.side == BLUE else "红方"
            act = "ban" if step.is_ban else "选人"
            lines.append(f"下一步：第 {self._cursor + 1} 手 · {side_name}（{who}）{act}")
        else:
            lines.append("下一步：B/P 已结束")
        return "\n".join(lines)

    def summary(self) -> dict:
        enc = self.encode()
        return {
            "game_index": self.game_index,
            "our_side": "blue" if self.our_side == BLUE else "red",
            "step": self._cursor,
            "bans": self.pool.names(self.bans()),
            "ours": self.pool.names(self.picks_of(self.our_side)),
            "opp": self.pool.names(self.picks_of(RED if self.our_side == BLUE else BLUE)),
            "encoded": enc,
            "locked": self.pool.names(self.series.locked_ids(self.our_side)) if self.series else [],
            "locked_opp": (
                self.pool.names(self.series.locked_ids(other(self.our_side))) if self.series else []
            ),
        }

    # ================= 便捷构造 =================
    def commit_to_series(self, winner: int | None = None) -> GameRecord:
        """把本局结果写回系列赛，推进全局 B/P 的消耗池。"""
        if self.series is None:
            raise BPCoachError("当前局面没有绑定系列赛上下文")
        bans_by_side: dict[int, list[int]] = {BLUE: [], RED: []}
        for _, hid, kind, side in self._history:
            if kind == BAN:
                bans_by_side[side].append(hid)
        rec = GameRecord(
            game_index=self.game_index,
            picks={BLUE: self.picks_of(BLUE), RED: self.picks_of(RED)},
            bans=bans_by_side,
            winner=winner,
        )
        self.series.commit(rec)
        return rec

    @classmethod
    def from_names(
        cls,
        pool: HeroPool,
        template: BPTemplate,
        names,
        our_side: int = BLUE,
        series: SeriesContext | None = None,
        **kwargs,
    ) -> "DraftState":
        """按模板顺序依次落子；名字可以是中文名或别名。"""
        state = cls(pool, template, our_side=our_side, series=series, **kwargs)
        for item in names:
            if isinstance(item, str) and ":" in item:
                kind, hero = item.split(":", 1)
                state.apply(hero.strip(), kind=kind.strip())
            else:
                state.apply(item)
        return state

    def __repr__(self) -> str:
        return (
            f"<DraftState step={self._cursor}/{self.total_steps} "
            f"our={self.meta()['our_side']} picks={len(self.picks_of(self.our_side))}>"
        )
