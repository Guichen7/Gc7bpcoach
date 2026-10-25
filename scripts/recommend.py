# -*- coding: utf-8 -*-
"""B/P 推荐入口：从命令行收一个 JSON 局面，输出 JSON 推荐。

用法
    python scripts/recommend.py '<局面JSON>'

局面 JSON 字段
    our_side     "blue" | "red"          必填，我方持边
    steps        ["盾山", ...]           本局已走的手，按顺序给英雄名或 id
                                         （ban/pick 由模板自动判定，不用标）
    prior_games  [{"picks_blue": [...], "picks_red": [...],
                   "winner": "blue"|"red"}, ...]
                                         可选。前几局的阵容与胜方（全局 B/P 锁定要用；
                                         打第 1 局时不传）
    depth        搜索深度，默认 3（1~2 更快但更短视）
    opp_risk     对手口径 0.0~1.0，默认 0（0=按概率加权，1=最坏情况）
    m3           "lineup"（默认）/ "team" / "state"

输出 JSON（ok=true 时）
    is_our_turn / step / rank_by / recommendation / candidates / baseline_p /
    threat_before / denial_blank / elapsed
ok=false 时只有 error 字段。
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bpcoach import SearchConfig, load_bundle  # noqa: E402
from bpcoach.errors import BPCoachError  # noqa: E402
from bpcoach.series import BLUE, RED, GameRecord, SeriesContext  # noqa: E402
from bpcoach.state import DraftState  # noqa: E402

BUNDLE = ROOT / "models" / "bpcoach_v0.5.0.bpc"
_SIDES = {"blue": BLUE, "red": RED}


def _fail(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    sys.exit(1)


def _cand(c) -> dict:
    return {
        "hero": c.name, "value": round(c.value, 4),
        "p_greedy": None if c.p_greedy is None else round(c.p_greedy, 4),
        "denial": round(c.denial, 4),
        "lane": c.lane, "m2_prob": round(c.policy, 4),
    }


def main() -> None:
    if len(sys.argv) != 2:
        _fail("用法：python scripts/recommend.py '<局面JSON>'（见文件头部说明）")
    try:
        req = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        _fail(f"JSON 解析失败：{e}")

    try:
        our = _SIDES[str(req["our_side"]).lower()]
    except (KeyError, KeyError):
        _fail("our_side 必须是 \"blue\" 或 \"red\"")

    search = load_bundle(
        BUNDLE, m3=str(req.get("m3", "lineup")),
        config=SearchConfig(depth=int(req.get("depth", 3)),
                            opp_risk=float(req.get("opp_risk", 0.0))))

    pool, tpl = search.pool, search.template
    series = SeriesContext(pool, global_lock=tpl.global_lock)
    for g in req.get("prior_games") or []:
        try:
            series.commit(GameRecord(
                game_index=series.game_index,
                picks={BLUE: [pool.index(h) for h in g["picks_blue"]],
                       RED: [pool.index(h) for h in g["picks_red"]]},
                bans={BLUE: [pool.index(h) for h in g.get("bans_blue", [])],
                      RED: [pool.index(h) for h in g.get("bans_red", [])]},
                winner=_SIDES[str(g["winner"]).lower()] if g.get("winner") else None,
            ))
        except BPCoachError as e:
            _fail(f"第 {series.game_index} 局记录有误：{e}")

    st = DraftState(pool=pool, template=tpl, our_side=our,
                    series=series if series.games else None)
    for h in req.get("steps") or []:
        try:
            st.apply(h)
        except BPCoachError as e:
            _fail(f"第 {st.step_index + 1} 手无法执行：{e}")

    rec = search.recommend(st)
    step = st.current_step
    out = {
        "ok": True,
        "is_our_turn": rec.is_our_turn,
        "step": None if step is None else {
            "index": step.index + 1,
            "kind": step.kind,
            "side": "blue" if step.side == BLUE else "red",
        },
        "rank_by": rec.rank_by,
        "degenerate": rec.degenerate,
        "recommendation": None if rec.best is None else _cand(rec.best),
        "candidates": [_cand(c) for c in rec.candidates[:5]],
        "baseline_p": round(rec.baseline_p, 4),
        "threat_before": None if rec.threat_before is None else round(rec.threat_before, 4),
        "denial_blank": rec.denial_blank,
        "opp_risk": rec.opp_risk,
        "elapsed_s": round(rec.elapsed, 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
