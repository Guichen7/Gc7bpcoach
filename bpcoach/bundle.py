# -*- coding: utf-8 -*-
"""单文件推理包（.bpc，本质是 zip）：把 config + models 打成一个文件，一行加载。

格式
    manifest.json   格式版本、构建时间、文件清单（sha256 + 字节数）、默认模板与 M3 变体
    config/...      heroes_s45.json / bp_templates.json / lane_usage.json
    models/...      markov（M2）、winrate 三个变体（M3）、mvp 及各自的元数据/指标

约定
    - 加载时先全量校验 sha256，任何字节被改都拒绝加载——宁可报错也不拿半个坏包跑。
    - 全程走内存（joblib 读 BytesIO），不向磁盘解压。
    - 包内不含代码，只含数据与模型参数；加载方仍是仓库里的 bpcoach。
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import zipfile
from datetime import datetime, timezone

from .errors import BPCoachError
from .heroes import HeroPool
from .template import BPTemplate

BUNDLE_FORMAT = "bpcoach-bundle"
BUNDLE_VERSION = 1

# 包内固定文件清单（仓库相对路径）。metrics 是评估留档，推理用不到但随包交付。
BUNDLE_FILES: tuple[str, ...] = (
    "config/heroes_s45.json",
    "config/bp_templates.json",
    "config/lane_usage.json",
    "models/markov_kpl2026.json",
    "models/markov_kpl2026_metrics.json",
    "models/winrate_kpl2026_lineup.joblib",
    "models/winrate_kpl2026_lineup.json",
    "models/winrate_kpl2026_team.joblib",
    "models/winrate_kpl2026_team.json",
    "models/winrate_kpl2026.joblib",
    "models/winrate_kpl2026.json",
    "models/winrate_kpl2026_metrics.json",
    "models/mvp_kpl2026.joblib",
    "models/mvp_kpl2026.json",
    "models/mvp_kpl2026_metrics.json",
)

# M3 变体：lineup 只看阵容（M4 默认口径）；team 混了队伍强度；state 是带上下文的版本。
M3_VARIANTS = {
    "lineup": "models/winrate_kpl2026_lineup.joblib",
    "team": "models/winrate_kpl2026_team.joblib",
    "state": "models/winrate_kpl2026.joblib",
}

_MANIFEST = "manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_bundle(out_path: str | pathlib.Path | None = None,
                 root: str | pathlib.Path | None = None) -> pathlib.Path:
    """把 config + models 打成 .bpc。返回产物路径。"""
    from . import __version__

    repo = pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parents[1]
    payloads: dict[str, bytes] = {}
    for rel in BUNDLE_FILES:
        p = repo / rel
        if not p.is_file():
            raise BPCoachError(f"打包缺文件：{rel}")
        payloads[rel] = p.read_bytes()

    markov = json.loads(payloads["models/markov_kpl2026.json"].decode("utf-8"))
    heroes = json.loads(payloads["config/heroes_s45.json"].decode("utf-8"))
    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "bpcoach_version": __version__,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool_size": int(heroes.get("size", len(heroes.get("heroes", [])))),
        "season": heroes.get("season", ""),
        "patch": heroes.get("patch", ""),
        "default_template_id": markov.get("template_id") or "kpl_2026_bo5_global",
        "default_m3": "lineup",
        "m3_variants": sorted(M3_VARIANTS),
        "files": {rel: {"sha256": _sha256(b), "size": len(b)}
                  for rel, b in payloads.items()},
    }

    out = pathlib.Path(out_path) if out_path else (
        repo / "dist" / f"bpcoach_v{__version__}.bpc")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for rel in BUNDLE_FILES:
            zf.writestr(rel, payloads[rel])
    return out


def read_manifest(path: str | pathlib.Path) -> dict:
    """只读清单不加载模型，用于交付前检查（版本、池大小、构建时间）。"""
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(_MANIFEST).decode("utf-8"))


def _verified_files(zf: zipfile.ZipFile) -> tuple[dict, dict[str, bytes]]:
    manifest = json.loads(zf.read(_MANIFEST).decode("utf-8"))
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BPCoachError(f"不是 bpcoach 推理包：{manifest.get('format')!r}")
    if manifest.get("version") != BUNDLE_VERSION:
        raise BPCoachError(
            f"推理包格式版本 {manifest.get('version')}，本代码只认 {BUNDLE_VERSION}")
    payloads: dict[str, bytes] = {}
    for rel, info in manifest["files"].items():
        data = zf.read(rel)
        if _sha256(data) != info["sha256"]:
            raise BPCoachError(f"推理包校验失败：{rel} 的 sha256 与清单不符")
        payloads[rel] = data
    return manifest, payloads


def load_bundle(path: str | pathlib.Path, *, template_id: str | None = None,
                m3: str = "lineup", **kwargs):
    """从 .bpc 一行加载：校验 → 池/模板/M2/M3 → DraftSearch。

    template_id  默认用包内清单的 default_template_id（训练时的模板）。
    m3           "lineup"（默认，M4 口径）/ "team" / "state"。
    **kwargs     透传给 DraftSearch（如 config=SearchConfig(...)）。
    """
    from .markov import TransitionModel
    from .search import DraftSearch
    from .winrate import WinRateModel

    if m3 not in M3_VARIANTS:
        raise BPCoachError(f"未知 M3 变体 {m3!r}，可选 {sorted(M3_VARIANTS)}")

    with zipfile.ZipFile(path) as zf:
        manifest, payloads = _verified_files(zf)

    pool = HeroPool.from_dict(
        json.loads(payloads["config/heroes_s45.json"].decode("utf-8")))
    pool.attach_lane_usage(
        json.loads(payloads["config/lane_usage.json"].decode("utf-8")))
    if pool.size != manifest["pool_size"]:
        raise BPCoachError(
            f"池大小 {pool.size} 与清单 {manifest['pool_size']} 不符")

    tid = template_id or manifest["default_template_id"]
    templates = json.loads(payloads["config/bp_templates.json"].decode("utf-8"))
    tpl_data = next((t for t in templates["templates"] if t["id"] == tid), None)
    if tpl_data is None:
        raise BPCoachError(f"包内模板里没有 {tid!r}")
    tpl = BPTemplate.from_dict(tpl_data)

    m2 = TransitionModel.from_dict(
        json.loads(payloads["models/markov_kpl2026.json"].decode("utf-8")), pool)

    import joblib
    blob = joblib.load(io.BytesIO(payloads[M3_VARIANTS[m3]]))
    m3_model = WinRateModel.from_blob(blob, pool)

    return DraftSearch(pool, m2, m3_model, template=tpl, **kwargs)
