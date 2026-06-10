#!/usr/bin/env python3
"""Compact bilingual visualizer for the two-stage World Cup 2026 simulation.

Purpose
-------
This script creates presentation-style graphics:

  1. group_stage_compact_{lang}.svg/.png
     Group-stage qualification cards.

  2. r32_compact_{lang}.svg/.png
     Round of 32 fixed by the selected group-stage scenario.

  3. knockout_r16_compact_{lang}.svg/.png
     R16 -> Final compact bracket, similar to a TV explainer bracket.

  4. compact_report_{lang}.html
     A simple HTML report embedding the compact figures.

Design choices
--------------
- No "card occurrence", "pair occurrence", or support-count column is shown.
- Group-stage qualifiers are emphasized visually.
- Qualification status is recomputed from position and advanced_third_groups.
- The compact knockout bracket starts from R16. R32 is shown separately.
- English and Japanese output are both supported.

Required input files
--------------------
The --input-dir should be an output directory produced by
two_stage_worldcup_2026.py and must contain:

  selected_group_scenarios.csv
  selected_group_scenario_standings.csv
  selected_group_scenario_r32.csv
  coherent_projected_brackets_all.csv
  weighted_stage_probabilities_topk.csv

Example
-------
python3 scripts/visualize_two_stage_worldcup_2026_compact.py \
  --input-dir outputs_two_stage_consensus_1000 \
  --output-dir outputs_two_stage_consensus_1000/visuals_compact \
  --language both \
  --scenario-limit 1
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

GROUPS = tuple("ABCDEFGHIJKL")
FONT_FAMILY = "'Noto Sans CJK JP','Hiragino Sans','Yu Gothic','Noto Sans','Arial',sans-serif"

I18N = {
    "en": {
        "html_title": "2026 World Cup compact visualization",
        "title": "2026 World Cup two-stage simulation",
        "subtitle": "Compact group-stage and knockout forecast",
        "group_title": "Group-stage qualification",
        "r32_title": "Round of 32",
        "bracket_title": "Knockout bracket from R16",
        "third_place": "Third-place match",
        "scenario": "Scenario",
        "annex": "Annex C",
        "third_groups": "advancing third-place groups",
        "advance": "QUALIFIED",
        "best_third": "BEST 3RD",
        "out": "OUT",
        "winner": "Winner",
        "win_prob": "Win prob.",
        "champion": "Champion",
        "champion_prob": "Champion probabilities",
        "weighted_champion_prob": "Weighted champion probabilities",
        "conditional_champion_prob": "Conditional champion probability",
        "final_win_prob": "Final win",
        "probability_note": "Match-box percentages are matchup-conditional win probabilities; the champion-box percentage is the scenario-conditional champion probability for the displayed winner.",
        "modal_score": "modal score",
        "constructed": "constructed consensus",
        "scenario_weight": "scenario weight",
        "open_ja": "Japanese version",
        "open_en": "English version",
    },
    "ja": {
        "html_title": "2026 W杯 compact 可視化",
        "title": "2026 W杯 2段階シミュレーション",
        "subtitle": "グループリーグとトーナメントの compact 予測",
        "group_title": "グループリーグ突破予測",
        "r32_title": "Round of 32",
        "bracket_title": "R16以降のトーナメント予測",
        "third_place": "3位決定戦",
        "scenario": "シナリオ",
        "annex": "Annex C",
        "third_groups": "3位通過組",
        "advance": "突破",
        "best_third": "3位通過",
        "out": "敗退",
        "winner": "勝者",
        "win_prob": "勝率",
        "champion": "優勝",
        "champion_prob": "優勝確率",
        "weighted_champion_prob": "重み付き優勝確率",
        "conditional_champion_prob": "条件付き優勝確率",
        "final_win_prob": "決勝勝率",
        "probability_note": "試合ボックスの%はその対戦が実現した条件での勝率，優勝ボックスの%は表示中シナリオ条件付き優勝確率です。",
        "modal_score": "最頻スコア",
        "constructed": "構成した consensus",
        "scenario_weight": "scenario 重み",
        "open_ja": "日本語版",
        "open_en": "英語版",
    },
}

TEAM_JA = {
    "Algeria": "アルジェリア",
    "Argentina": "アルゼンチン",
    "Australia": "オーストラリア",
    "Austria": "オーストリア",
    "Belgium": "ベルギー",
    "Bosnia and Herzegovina": "ボスニア・ヘルツェゴビナ",
    "Brazil": "ブラジル",
    "Canada": "カナダ",
    "Cape Verde": "カーボベルデ",
    "Colombia": "コロンビア",
    "Croatia": "クロアチア",
    "Curaçao": "キュラソー",
    "Czech Republic": "チェコ",
    "Dem. Rep. of Congo": "コンゴ民主共和国",
    "Ecuador": "エクアドル",
    "Egypt": "エジプト",
    "England": "イングランド",
    "France": "フランス",
    "Germany": "ドイツ",
    "Ghana": "ガーナ",
    "Haiti": "ハイチ",
    "Iran": "イラン",
    "Iraq": "イラク",
    "Ivory Coast": "コートジボワール",
    "Japan": "日本",
    "Jordan": "ヨルダン",
    "Mexico": "メキシコ",
    "Morocco": "モロッコ",
    "Netherlands": "オランダ",
    "New Zealand": "ニュージーランド",
    "Norway": "ノルウェー",
    "Panama": "パナマ",
    "Paraguay": "パラグアイ",
    "Portugal": "ポルトガル",
    "Qatar": "カタール",
    "Saudi Arabia": "サウジアラビア",
    "Scotland": "スコットランド",
    "Senegal": "セネガル",
    "South Africa": "南アフリカ",
    "South Korea": "韓国",
    "Spain": "スペイン",
    "Sweden": "スウェーデン",
    "Switzerland": "スイス",
    "Tunisia": "チュニジア",
    "Turkey": "トルコ",
    "United States": "アメリカ",
    "Uruguay": "ウルグアイ",
    "Uzbekistan": "ウズベキスタン",
}

TEAM_FLAG = {
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Cape Verde": "🇨🇻",
    "Colombia": "🇨🇴",
    "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼",
    "Czech Republic": "🇨🇿",
    "Dem. Rep. of Congo": "🇨🇩",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Haiti": "🇭🇹",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Ivory Coast": "🇨🇮",
    "Japan": "🇯🇵",
    "Jordan": "🇯🇴",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴",
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}

TEAM_CODE = {
    "Algeria": "ALG", "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Brazil": "BRA", "Canada": "CAN",
    "Cape Verde": "CPV", "Colombia": "COL", "Croatia": "CRO", "Curaçao": "CUW",
    "Czech Republic": "CZE", "Dem. Rep. of Congo": "COD", "Ecuador": "ECU",
    "Egypt": "EGY", "England": "ENG", "France": "FRA", "Germany": "GER",
    "Ghana": "GHA", "Haiti": "HAI", "Iran": "IRN", "Iraq": "IRQ",
    "Ivory Coast": "CIV", "Japan": "JPN", "Jordan": "JOR", "Mexico": "MEX",
    "Morocco": "MAR", "Netherlands": "NED", "New Zealand": "NZL", "Norway": "NOR",
    "Panama": "PAN", "Paraguay": "PAR", "Portugal": "POR", "Qatar": "QAT",
    "Saudi Arabia": "KSA", "Scotland": "SCO", "Senegal": "SEN", "South Africa": "RSA",
    "South Korea": "KOR", "Spain": "ESP", "Sweden": "SWE", "Switzerland": "SUI",
    "Tunisia": "TUN", "Turkey": "TUR", "United States": "USA", "Uruguay": "URU",
    "Uzbekistan": "UZB",
}

def svg_flag(team: str, x: float, y: float, w: float = 36, h: float = 24) -> str:
    """Draw a small simplified flag as SVG shapes.

    Emoji flags often render as tofu boxes in CairoSVG or some browsers.  These
    vector mini-flags are intentionally simplified, but stable in SVG/PNG.
    """
    def rect(rx, ry, rw, rh, fill, stroke="none", sw=0):
        return f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    def circ(cx, cy, r, fill, stroke="none", sw=0):
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    def poly(points, fill, stroke="none", sw=0):
        pts = " ".join(f"{px},{py}" for px, py in points)
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    def text(tx, ty, value, size, fill="#111", weight="900", anchor="middle"):
        return (
            f'<text x="{tx}" y="{ty}" font-family="{FONT_FAMILY}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
            f'{esc(value)}</text>'
        )

    out = [rect(x, y, w, h, "#fff", "#111", 0.8)]
    t = team

    # Generic helpers
    def hbands(cols):
        hh = h / len(cols)
        return "".join(rect(x, y + i * hh, w, hh + 0.3, c) for i, c in enumerate(cols))

    def vbands(cols):
        ww = w / len(cols)
        return "".join(rect(x + i * ww, y, ww + 0.3, h, c) for i, c in enumerate(cols))

    if t == "Germany":
        out.append(hbands(["#000000", "#dd0000", "#ffce00"]))
    elif t == "France":
        out.append(vbands(["#002395", "#ffffff", "#ed2939"]))
    elif t == "Netherlands":
        out.append(hbands(["#ae1c28", "#ffffff", "#21468b"]))
    elif t == "Spain":
        out.append(hbands(["#aa151b", "#f1bf00", "#aa151b"]))
    elif t == "Argentina":
        out.append(hbands(["#75aadb", "#ffffff", "#75aadb"]))
        out.append(circ(x + w/2, y + h/2, h*0.10, "#f6b40e"))
    elif t == "Brazil":
        out.append(rect(x, y, w, h, "#009b3a"))
        out.append(poly([(x+w/2, y+3), (x+w-4, y+h/2), (x+w/2, y+h-3), (x+4, y+h/2)], "#ffdf00"))
        out.append(circ(x+w/2, y+h/2, h*0.23, "#002776"))
    elif t == "Japan":
        out.append(rect(x, y, w, h, "#ffffff"))
        out.append(circ(x+w/2, y+h/2, h*0.28, "#bc002d"))
    elif t == "Switzerland":
        out.append(rect(x, y, w, h, "#d52b1e"))
        out.append(rect(x+w*0.42, y+h*0.22, w*0.16, h*0.56, "#ffffff"))
        out.append(rect(x+w*0.25, y+h*0.40, w*0.50, h*0.18, "#ffffff"))
    elif t == "England":
        out.append(rect(x, y, w, h, "#ffffff"))
        out.append(rect(x+w*0.43, y, w*0.14, h, "#cf142b"))
        out.append(rect(x, y+h*0.42, w, h*0.16, "#cf142b"))
    elif t == "Scotland":
        out.append(rect(x, y, w, h, "#0065bd"))
        out.append(poly([(x, y), (x+5, y), (x+w, y+h-5), (x+w, y+h), (x+w-5, y+h), (x, y+5)], "#ffffff"))
        out.append(poly([(x+w, y), (x+w, y+5), (x+5, y+h), (x, y+h), (x, y+h-5), (x+w-5, y)], "#ffffff"))
    elif t == "Mexico":
        out.append(vbands(["#006847", "#ffffff", "#ce1126"]))
        out.append(circ(x+w/2, y+h/2, h*0.08, "#8c6b25"))
    elif t == "United States":
        stripe_h = h / 7
        for i in range(7):
            out.append(rect(x, y+i*stripe_h, w, stripe_h+0.2, "#b22234" if i % 2 == 0 else "#ffffff"))
        out.append(rect(x, y, w*0.42, stripe_h*4, "#3c3b6e"))
    elif t == "Canada":
        out.append(vbands(["#d52b1e", "#ffffff", "#d52b1e"]))
        out.append(text(x+w/2, y+h*0.68, "✦", h*0.55, "#d52b1e"))
    elif t == "Portugal":
        out.append(vbands(["#006600", "#ff0000"]))
        out.append(circ(x+w*0.42, y+h/2, h*0.13, "#ffcc00"))
    elif t == "Colombia":
        out.append(hbands(["#fcd116", "#003893", "#ce1126"]))
    elif t == "Ecuador":
        out.append(hbands(["#ffdd00", "#034ea2", "#ed1c24"]))
        out.append(circ(x+w/2, y+h/2, h*0.07, "#8c6b25"))
    elif t == "Uruguay":
        for i in range(6):
            out.append(rect(x, y+i*h/6, w, h/6+0.2, "#ffffff" if i % 2 == 0 else "#75aadb"))
        out.append(rect(x, y, w*0.32, h*0.50, "#ffffff"))
        out.append(circ(x+w*0.16, y+h*0.25, h*0.10, "#fcd116"))
    elif t == "Paraguay":
        out.append(hbands(["#d52b1e", "#ffffff", "#0038a8"]))
    elif t == "Belgium":
        out.append(vbands(["#000000", "#fae042", "#ed2939"]))
    elif t == "Croatia":
        out.append(hbands(["#ff0000", "#ffffff", "#171796"]))
        out.append(rect(x+w*0.43, y+h*0.34, w*0.14, h*0.32, "#ff0000"))
    elif t == "Austria":
        out.append(hbands(["#ed2939", "#ffffff", "#ed2939"]))
    elif t == "Czech Republic":
        out.append(hbands(["#ffffff", "#d7141a"]))
        out.append(poly([(x, y), (x+w*0.48, y+h/2), (x, y+h)], "#11457e"))
    elif t == "Bosnia and Herzegovina":
        out.append(rect(x, y, w, h, "#002395"))
        out.append(poly([(x+w*0.43, y), (x+w*0.75, y+h), (x+w*0.25, y+h)], "#fecb00"))
    elif t == "Sweden":
        out.append(rect(x, y, w, h, "#006aa7"))
        out.append(rect(x+w*0.30, y, w*0.13, h, "#fecc00"))
        out.append(rect(x, y+h*0.42, w, h*0.16, "#fecc00"))
    elif t == "Norway":
        out.append(rect(x, y, w, h, "#ba0c2f"))
        out.append(rect(x+w*0.27, y, w*0.22, h, "#ffffff"))
        out.append(rect(x, y+h*0.38, w, h*0.24, "#ffffff"))
        out.append(rect(x+w*0.32, y, w*0.11, h, "#00205b"))
        out.append(rect(x, y+h*0.44, w, h*0.12, "#00205b"))
    elif t == "Turkey":
        out.append(rect(x, y, w, h, "#e30a17"))
        out.append(circ(x+w*0.40, y+h/2, h*0.22, "#ffffff"))
        out.append(circ(x+w*0.47, y+h/2, h*0.18, "#e30a17"))
        out.append(text(x+w*0.62, y+h*0.67, "★", h*0.30, "#ffffff"))
    elif t == "Morocco":
        out.append(rect(x, y, w, h, "#c1272d"))
        out.append(text(x+w/2, y+h*0.67, "★", h*0.34, "#006233"))
    elif t == "Senegal":
        out.append(vbands(["#00853f", "#fdef42", "#e31b23"]))
        out.append(text(x+w/2, y+h*0.67, "★", h*0.30, "#00853f"))
    elif t == "Ghana":
        out.append(hbands(["#ce1126", "#fcd116", "#006b3f"]))
        out.append(text(x+w/2, y+h*0.67, "★", h*0.30, "#000000"))
    elif t == "Ivory Coast":
        out.append(vbands(["#f77f00", "#ffffff", "#009e60"]))
    elif t == "South Africa":
        out.append(hbands(["#de3831", "#ffffff", "#002395"]))
        out.append(poly([(x, y), (x+w*0.45, y+h/2), (x, y+h)], "#007a4d"))
        out.append(poly([(x, y+2), (x+w*0.25, y+h/2), (x, y+h-2)], "#000000"))
    elif t == "Cape Verde":
        out.append(rect(x, y, w, h, "#003893"))
        out.append(rect(x, y+h*0.55, w, h*0.10, "#ffffff"))
        out.append(rect(x, y+h*0.64, w, h*0.06, "#cf2027"))
        out.append(circ(x+w*0.30, y+h*0.47, h*0.07, "#ffcc00"))
    elif t == "Tunisia":
        out.append(rect(x, y, w, h, "#e70013"))
        out.append(circ(x+w/2, y+h/2, h*0.25, "#ffffff"))
        out.append(text(x+w/2, y+h*0.64, "★", h*0.25, "#e70013"))
    elif t == "Algeria":
        out.append(vbands(["#006233", "#ffffff"]))
        out.append(text(x+w*0.55, y+h*0.64, "★", h*0.25, "#d21034"))
    elif t == "Egypt":
        out.append(hbands(["#ce1126", "#ffffff", "#000000"]))
        out.append(circ(x+w/2, y+h/2, h*0.06, "#c09300"))
    elif t == "Iran":
        out.append(hbands(["#239f40", "#ffffff", "#da0000"]))
    elif t == "Iraq":
        out.append(hbands(["#ce1126", "#ffffff", "#000000"]))
    elif t == "Saudi Arabia":
        out.append(rect(x, y, w, h, "#006c35"))
        out.append(rect(x+w*0.20, y+h*0.55, w*0.60, h*0.08, "#ffffff"))
    elif t == "Qatar":
        out.append(vbands(["#ffffff", "#8a1538"]))
    elif t == "Jordan":
        out.append(hbands(["#000000", "#ffffff", "#007a3d"]))
        out.append(poly([(x, y), (x+w*0.42, y+h/2), (x, y+h)], "#ce1126"))
    elif t == "Australia":
        out.append(rect(x, y, w, h, "#00008b"))
        out.append(text(x+w*0.73, y+h*0.68, "★", h*0.30, "#ffffff"))
    elif t == "New Zealand":
        out.append(rect(x, y, w, h, "#00247d"))
        out.append(text(x+w*0.70, y+h*0.68, "★", h*0.30, "#cc142b"))
    elif t == "South Korea":
        out.append(rect(x, y, w, h, "#ffffff"))
        out.append(circ(x+w/2, y+h/2, h*0.20, "#cd2e3a"))
        out.append(circ(x+w/2, y+h/2+h*0.06, h*0.15, "#0047a0"))
    elif t == "Japan":
        pass
    elif t == "Panama":
        out.append(rect(x, y, w/2, h/2, "#ffffff"))
        out.append(rect(x+w/2, y, w/2, h/2, "#d21034"))
        out.append(rect(x, y+h/2, w/2, h/2, "#005293"))
        out.append(rect(x+w/2, y+h/2, w/2, h/2, "#ffffff"))
        out.append(text(x+w*0.25, y+h*0.38, "★", h*0.22, "#005293"))
        out.append(text(x+w*0.75, y+h*0.88, "★", h*0.22, "#d21034"))
    elif t == "Haiti":
        out.append(hbands(["#00209f", "#d21034"]))
    elif t == "Curaçao":
        out.append(rect(x, y, w, h, "#002b7f"))
        out.append(rect(x, y+h*0.62, w, h*0.11, "#f9e814"))
        out.append(text(x+w*0.25, y+h*0.45, "★", h*0.20, "#ffffff"))
    elif t == "Dem. Rep. of Congo":
        out.append(rect(x, y, w, h, "#00a3e0"))
        out.append(poly([(x, y+h), (x+w, y)], "#f7d618"))
        out.append(poly([(x, y+h*0.88), (x+w*0.88, y)], "#ce1021"))
        out.append(text(x+w*0.20, y+h*0.42, "★", h*0.24, "#f7d618"))
    elif t == "Uzbekistan":
        out.append(hbands(["#1eb5e5", "#ffffff", "#009739"]))
        out.append(rect(x, y+h/3-1, w, 1.6, "#ce1126"))
        out.append(rect(x, y+2*h/3-1, w, 1.6, "#ce1126"))
    else:
        out.append(rect(x, y, w, h, "#f0f0f0"))
        out.append(text(x+w/2, y+h*0.67, TEAM_CODE.get(t, t[:3].upper()), h*0.36, "#111"))

    out.append(rect(x, y, w, h, "none", "#111", 0.8))
    return "".join(out)



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def as_float(x: object, default: float = float("nan")) -> float:
    try:
        return float(str(x))
    except Exception:
        return default


def as_int(x: object, default: int = -1) -> int:
    try:
        return int(str(x))
    except Exception:
        return default


def pct(x: object, decimals: int = 1) -> str:
    v = as_float(x)
    if math.isnan(v):
        return ""
    return f"{100.0 * v:.{decimals}f}%"


def tr(lang: str, key: str) -> str:
    return I18N[lang][key]


def flag(team: str) -> str:
    return TEAM_FLAG.get(team, "")


def team_label(team: str, lang: str) -> str:
    if lang == "ja":
        return TEAM_JA.get(team, team)
    return team


def parse_third_groups(scenario: Mapping[str, str]) -> set[str]:
    raw = scenario.get("advanced_third_groups", "") or ""
    return {g for g in re.split(r"[/,\s]+", raw) if g in GROUPS}


def qualification_status(row: Mapping[str, str], scenario: Mapping[str, str]) -> str:
    """Return auto, best_third, or out.

    The status is recomputed from position and advanced_third_groups. This avoids
    visual errors when an older CSV contains an overly permissive status column.
    """
    pos = as_int(row.get("position", ""))
    group = str(row.get("group", ""))
    third_groups = parse_third_groups(scenario)
    if pos <= 2:
        return "auto"
    if pos == 3 and group in third_groups:
        return "best_third"
    return "out"


def wrap_label(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def scenario_label(s: Mapping[str, str], lang: str) -> str:
    gp = pct(s.get("group_probability", ""))
    if gp:
        return gp
    return tr(lang, "constructed")


def svg_text(x, y, text, size=16, weight="700", fill="#111", anchor="start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT_FAMILY}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f'{esc(text)}</text>'
    )


def load_outputs(input_dir: Path) -> dict:
    data = {
        "selected": read_csv(input_dir / "selected_group_scenarios.csv"),
        "standings": read_csv(input_dir / "selected_group_scenario_standings.csv"),
        "r32": read_csv(input_dir / "selected_group_scenario_r32.csv"),
        "brackets": read_csv(input_dir / "coherent_projected_brackets_all.csv"),
    }
    weighted_path = input_dir / "weighted_stage_probabilities_topk.csv"
    data["weighted"] = read_csv(weighted_path) if weighted_path.exists() else []

    scenario_champs_path = input_dir / "scenario_champion_probabilities.csv"
    data["scenario_champs"] = read_csv(scenario_champs_path) if scenario_champs_path.exists() else []

    scenario_team_probs = {}
    for row in data["selected"]:
        sid = row.get("scenario_id", "")
        if not sid:
            continue
        probs_path = input_dir / f"{sid}_conditional_team_stage_probabilities.csv"
        if probs_path.exists():
            scenario_team_probs[sid] = read_csv(probs_path)
    data["scenario_team_probs"] = scenario_team_probs
    return data


def scenario_champion_probability(
    data: Mapping[str, object],
    sid: str,
    team: str,
) -> Optional[float]:
    """Return P(team champion | displayed scenario), when available.

    This is intentionally different from
    winner_conditional_probability_given_pair in the final match row.  The latter
    is P(win final | this exact final pairing), while this function returns the
    scenario-conditional tournament champion probability for the team.
    """
    if not team:
        return None

    scenario_team_probs = data.get("scenario_team_probs", {})
    if isinstance(scenario_team_probs, Mapping):
        rows = scenario_team_probs.get(sid, [])
        if isinstance(rows, Sequence):
            for row in rows:
                if isinstance(row, Mapping) and row.get("team") == team:
                    val = as_float(row.get("champion", ""), float("nan"))
                    if not math.isnan(val):
                        return val

    scenario_champs = data.get("scenario_champs", [])
    if isinstance(scenario_champs, Sequence):
        for row in scenario_champs:
            if (
                isinstance(row, Mapping)
                and row.get("scenario_id") == sid
                and row.get("team") == team
            ):
                val = as_float(row.get("champion_conditional_probability", ""), float("nan"))
                if not math.isnan(val):
                    return val

    # Fallback for old one-scenario outputs.  Do not use this for multi-scenario
    # top-k reports because it would mix a weighted marginal probability into a
    # scenario-specific bracket.
    selected = data.get("selected", [])
    weighted = data.get("weighted", [])
    if isinstance(selected, Sequence) and len(selected) == 1 and isinstance(weighted, Sequence):
        for row in weighted:
            if isinstance(row, Mapping) and row.get("team") == team:
                val = as_float(row.get("champion", ""), float("nan"))
                if not math.isnan(val):
                    return val
    return None


def champion_probability_label(data: Mapping[str, object], sid: str, team: str, lang: str) -> str:
    val = scenario_champion_probability(data, sid, team)
    if val is None:
        return ""
    return f"{tr(lang, 'conditional_champion_prob')}: {pct(val)}"


def build_indices(data: Mapping[str, Sequence[Mapping[str, str]]]):
    scenarios = {r["scenario_id"]: dict(r) for r in data["selected"]}

    standings = defaultdict(lambda: defaultdict(list))
    for row in data["standings"]:
        standings[row["scenario_id"]][row["group"]].append(dict(row))
    for sid in standings:
        for g in standings[sid]:
            standings[sid][g].sort(key=lambda r: as_int(r.get("position", "")))

    r32 = defaultdict(list)
    for row in data["r32"]:
        r32[row["scenario_id"]].append(dict(row))
    for sid in r32:
        r32[sid].sort(key=lambda r: as_int(r.get("match_no", "")))

    brackets = defaultdict(list)
    for row in data["brackets"]:
        brackets[row["scenario_id"]].append(dict(row))
    for sid in brackets:
        brackets[sid].sort(key=lambda r: as_int(r.get("match_no", "")))

    return scenarios, standings, r32, brackets


def status_style(status: str) -> Tuple[str, str, str]:
    # fill, stroke, text
    if status == "auto":
        return "#2fbf71", "#0b6b3d", "#ffffff"
    if status == "best_third":
        return "#f3c14b", "#9a6b00", "#1b1600"
    return "#e7ecf6", "#8a96aa", "#2b3448"


def row_style(is_winner: bool, final: bool = False) -> Tuple[str, str, str]:
    if final and is_winner:
        return "#f4c542", "#916900", "#1b1600"
    if is_winner:
        return "#c9f5db", "#15834d", "#102818"
    return "#f5f7fc", "#111111", "#111111"


# ---------------------------------------------------------------------
# Group-stage compact SVG
# ---------------------------------------------------------------------

def write_group_stage_svg(path: Path, data: Mapping[str, Sequence[Mapping[str, str]]], sid: str, lang: str) -> None:
    scenarios, standings, _, _ = build_indices(data)
    scenario = scenarios[sid]

    W, H = 1800, 1160
    bg = "#dce5f7"
    card_fill = "#f6f8ff"
    line = "#111111"
    muted = "#526077"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{bg}"/>',
        f'<circle cx="38" cy="38" r="30" fill="none" stroke="#49a6aa" stroke-width="6" opacity=".9"/>',
        svg_text(70, 70, tr(lang, "title"), 42, "900", "#111"),
        svg_text(70, 108, tr(lang, "group_title"), 24, "800", muted),
        svg_text(70, 138, f"{tr(lang,'scenario')}: {sid} / {tr(lang,'annex')}: {scenario.get('annex_c_option','')} / {tr(lang,'third_groups')}: {scenario.get('advanced_third_groups','')}", 17, "700", muted),
    ]

    card_w, card_h = 410, 285
    start_x, start_y = 70, 185
    gap_x, gap_y = 35, 34

    for idx, g in enumerate(GROUPS):
        col = idx % 4
        row = idx // 4
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)

        svg += [
            f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="10" fill="{card_fill}" stroke="{line}" stroke-width="3"/>',
            f'<rect x="{x}" y="{y}" width="{card_w}" height="45" rx="10" fill="#cad5ea" stroke="{line}" stroke-width="3"/>',
            f'<rect x="{x}" y="{y+25}" width="{card_w}" height="20" fill="#cad5ea"/>',
            svg_text(x + 18, y + 31, f"Group {g}", 22, "900", "#111"),
        ]

        rows = standings[sid].get(g, [])
        for i, r in enumerate(rows[:4]):
            status = qualification_status(r, scenario)
            fill, stroke, tc = status_style(status)
            tag = tr(lang, "advance") if status == "auto" else tr(lang, "best_third") if status == "best_third" else tr(lang, "out")
            ry = y + 62 + i * 52
            team = r.get("team", "")
            name = wrap_label(team_label(team, lang), 16 if lang == "ja" else 24)
            svg += [
                f'<rect x="{x+16}" y="{ry}" width="{card_w-32}" height="42" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                svg_text(x + 32, ry + 28, str(r.get("position", "")), 18, "900", tc),
                svg_flag(team, x + 66, ry + 9, 34, 23),
                svg_text(x + 110, ry + 28, name, 19, "900", tc),
                svg_text(x + card_w - 22, ry + 27, tag, 12, "900", tc, "end"),
            ]

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


# ---------------------------------------------------------------------
# R32 compact SVG
# ---------------------------------------------------------------------

def write_r32_svg(path: Path, data: Mapping[str, Sequence[Mapping[str, str]]], sid: str, lang: str) -> None:
    scenarios, _, r32_rows, bracket_rows = build_indices(data)
    scenario = scenarios[sid]
    r32 = r32_rows[sid]
    r32_forecast = {r["match_no"]: r for r in bracket_rows[sid] if r.get("round") == "R32"}

    W, H = 1800, 1120
    bg = "#dce5f7"
    line = "#111111"
    muted = "#526077"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{bg}"/>',
        f'<circle cx="38" cy="38" r="30" fill="none" stroke="#49a6aa" stroke-width="6" opacity=".9"/>',
        svg_text(70, 70, tr(lang, "title"), 42, "900", "#111"),
        svg_text(70, 108, tr(lang, "r32_title"), 24, "800", muted),
        svg_text(70, 138, f"{tr(lang,'scenario')}: {sid} / {tr(lang,'annex')}: {scenario.get('annex_c_option','')}", 17, "700", muted),
    ]

    card_w, card_h = 405, 190
    start_x, start_y = 70, 185
    gap_x, gap_y = 35, 36

    for idx, m in enumerate(r32):
        col = idx % 4
        row = idx // 4
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)
        match_no = m.get("match_no", "")
        forecast = r32_forecast.get(match_no, {})
        winner = forecast.get("projected_winner", "")
        prob = pct(forecast.get("winner_conditional_probability_given_pair", ""))

        svg += [
            f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="10" fill="#f6f8ff" stroke="{line}" stroke-width="3"/>',
            svg_text(x + 16, y + 28, f"M{match_no}", 18, "900", muted),
        ]

        for i, team in enumerate([m.get("team1", ""), m.get("team2", "")]):
            is_win = team == winner
            fill, stroke, tc = row_style(is_win)
            ry = y + 48 + i * 52
            name = wrap_label(team_label(team, lang), 16 if lang == "ja" else 24)
            svg += [
                f'<rect x="{x+18}" y="{ry}" width="{card_w-36}" height="42" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                svg_flag(team, x + 34, ry + 9, 36, 24),
                svg_text(x + 80, ry + 28, name, 20, "900" if is_win else "750", tc),
            ]
            if is_win:
                svg.append(svg_text(x + card_w - 24, ry + 28, prob, 16, "900", tc, "end"))

        svg.append(svg_text(x + 18, y + card_h - 18, f"{tr(lang,'winner')}: {team_label(winner, lang)}", 13, "900", "#15834d"))

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


# ---------------------------------------------------------------------
# R16+ bracket compact SVG
# ---------------------------------------------------------------------

def match_by_no(rows: Sequence[Mapping[str, str]]) -> Dict[int, Mapping[str, str]]:
    return {as_int(r.get("match_no", "")): r for r in rows}


def box_center(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    return x + w / 2, y + h / 2


def draw_connector(svg: List[str], child: Tuple[float, float, float, float], parent: Tuple[float, float, float, float], side: str) -> None:
    x, y, w, h = child
    px, py, pw, ph = parent
    cy = y + h / 2
    pyc = py + ph / 2
    if side == "left":
        x1 = x + w
        x2 = px
        mid = (x1 + x2) / 2
        d = f"M {x1} {cy} H {mid} V {pyc} H {x2}"
    else:
        x1 = x
        x2 = px + pw
        mid = (x1 + x2) / 2
        d = f"M {x1} {cy} H {mid} V {pyc} H {x2}"
    svg.append(f'<path d="{d}" fill="none" stroke="#111" stroke-width="4" stroke-linecap="square"/>')


def draw_loser_connector(
    svg: List[str],
    semifinal_box: Tuple[float, float, float, float],
    third_place_box: Tuple[float, float, float, float],
    side: str,
) -> None:
    """Draw a dashed path from a semifinal loser to the third-place match."""
    x, y, w, h = semifinal_box
    tx, ty, tw, th = third_place_box
    start_x = x + w / 2
    start_y = y + h
    end_x = tx + (tw * 0.35 if side == "left" else tw * 0.65)
    end_y = ty
    mid_y = (start_y + end_y) / 2
    d = f"M {start_x} {start_y} V {mid_y} H {end_x} V {end_y}"
    svg.append(
        f'<path d="{d}" fill="none" stroke="#7a6424" stroke-width="3.2" '
        f'stroke-dasharray="9 8" stroke-linecap="round" opacity=".88"/>'
    )

def draw_team_match_box(
    svg: List[str],
    x: float,
    y: float,
    w: float,
    h: float,
    row: Mapping[str, str],
    lang: str,
    final: bool = False,
) -> None:
    team1 = row.get("team1", "")
    team2 = row.get("team2", "")
    winner = row.get("projected_winner", "")
    prob = pct(row.get("winner_conditional_probability_given_pair", ""))

    svg += [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#eef3ff" stroke="#111" stroke-width="4"/>',
        f'<line x1="{x}" y1="{y+h/2}" x2="{x+w}" y2="{y+h/2}" stroke="#111" stroke-width="3"/>',
    ]

    for i, team in enumerate([team1, team2]):
        is_win = team == winner
        fill, stroke, tc = row_style(is_win, final=final)
        ry = y + i * (h / 2)
        svg.append(f'<rect x="{x+3}" y="{ry+3}" width="{w-6}" height="{h/2-6}" fill="{fill}" stroke="none"/>')
        name = wrap_label(team_label(team, lang), 11 if lang == "ja" else 15)
        svg.append(svg_flag(team, x + 14, ry + h / 4 - 10, 36, 24))
        svg.append(svg_text(x + 60, ry + h / 4 + 8, name, 22 if lang == "ja" else 19, "900" if is_win else "760", tc))
        if is_win:
            prob_text = f"{tr(lang, 'final_win_prob')} {prob}" if final and prob else prob
            svg.append(svg_text(x + w - 14, ry + h / 4 + 8, prob_text, 13 if final else 14, "900", tc, "end"))

def write_r16_bracket_svg(path: Path, data: Mapping[str, Sequence[Mapping[str, str]]], sid: str, lang: str) -> None:
    """Write a readable R16-to-final bracket.

    The previous compact layout placed the two semifinal boxes and the final
    match box on almost the same y-coordinate.  Because the boxes are wide, the
    central three boxes could visually overlap.  This layout separates the
    winner path vertically: semifinal boxes are kept in the middle row, the
    final match is moved upward, and the champion box is placed above the final.
    """
    scenarios, _, _, brackets = build_indices(data)
    scenario = scenarios[sid]
    rows = match_by_no(brackets[sid])

    W, H = 1880, 990
    bg = "#dce5f7"
    muted = "#526077"
    panel = "#f6f8ff"
    line = "#111"

    def round_label(code: str) -> str:
        ja = {"R16": "R16", "QF": "準々決勝", "SF": "準決勝", "Final": "決勝", "Third place": "3位決定戦", "Champion": "優勝"}
        en = {"R16": "R16", "QF": "Quarterfinal", "SF": "Semifinal", "Final": "Final", "Third place": "Third-place match", "Champion": "Champion"}
        return (ja if lang == "ja" else en)[code]

    def header(x: float, y: float, w: float, label: str) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="15" fill="{panel}" '
            f'stroke="{line}" stroke-width="3"/>'
            + svg_text(x + w / 2, y + 21, label, 15, "900", "#111", "middle")
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{bg}"/>',
        f'<circle cx="38" cy="38" r="30" fill="none" stroke="#49a6aa" stroke-width="6" opacity=".9"/>',
        svg_text(70, 70, tr(lang, "title"), 42, "900", "#111"),
        svg_text(70, 108, tr(lang, "bracket_title"), 24, "800", muted),
        svg_text(70, 138, f"{tr(lang,'scenario')}: {sid} / {tr(lang,'annex')}: {scenario.get('annex_c_option','')}", 17, "700", muted),
        svg_text(70, 166, tr(lang, "probability_note"), 14, "700", muted),
    ]

    # Readable broadcast-style coordinates.  Horizontal symmetry is preserved,
    # but the final is lifted so that it no longer overlaps the semifinal boxes.
    bw, bh = 280, 88
    final_w, final_h = 320, 98
    champ_w, champ_h = 320, 62
    coords: Dict[int, Tuple[float, float, float, float]] = {}

    x16_l, xqf_l, xsf_l = 45, 355, 620
    xsf_r, xqf_r, x16_r = 980, 1245, 1555
    y16_l = {89: 240, 90: 360, 93: 670, 94: 790}
    y16_r = {91: 240, 92: 360, 95: 670, 96: 790}

    for m, y in y16_l.items():
        coords[m] = (x16_l, y, bw, bh)
    for m, y in y16_r.items():
        coords[m] = (x16_r, y, bw, bh)

    coords[97] = (xqf_l, 300, bw, bh)
    coords[98] = (xqf_l, 730, bw, bh)
    coords[99] = (xqf_r, 300, bw, bh)
    coords[100] = (xqf_r, 730, bw, bh)
    coords[101] = (xsf_l, 515, bw, bh)
    coords[102] = (xsf_r, 515, bw, bh)

    final_box = ((W - final_w) / 2, 250, final_w, final_h)
    champ_box = ((W - champ_w) / 2, 156, champ_w, champ_h)
    third_place_box = ((W - final_w) / 2, 705, final_w, final_h)

    # Round headers are drawn after connectors, so connector paths never cover
    # the header labels.
    headers = [
        header(x16_l, 198, bw, round_label("R16")),
        header(xqf_l, 258, bw, round_label("QF")),
        header(xsf_l, 473, bw, round_label("SF")),
        header(xsf_r, 473, bw, round_label("SF")),
        header(xqf_r, 258, bw, round_label("QF")),
        header(x16_r, 198, bw, round_label("R16")),
        header(final_box[0], final_box[1] - 40, final_w, round_label("Final")),
        header(third_place_box[0], third_place_box[1] - 40, final_w, round_label("Third place")),
    ]

    # Connectors first, so boxes and headers are drawn on top.
    for child, parent in [(89, 97), (90, 97), (93, 98), (94, 98), (97, 101), (98, 101)]:
        draw_connector(svg, coords[child], coords[parent], "left")
    for child, parent in [(91, 99), (92, 99), (95, 100), (96, 100), (99, 102), (100, 102)]:
        draw_connector(svg, coords[child], coords[parent], "right")

    # Semifinals to final, and semifinal losers to third-place match.
    draw_connector(svg, coords[101], final_box, "left")
    draw_connector(svg, coords[102], final_box, "right")
    draw_loser_connector(svg, coords[101], third_place_box, "left")
    draw_loser_connector(svg, coords[102], third_place_box, "right")

    # Final to champion.
    fx, fy, fw, fh = final_box
    cx, cy, cw, ch = champ_box
    svg.append(f'<path d="M {fx+fw/2} {fy} V {cy+ch}" fill="none" stroke="{line}" stroke-width="4"/>')
    svg.extend(headers)

    # Match boxes.
    for m in [89, 90, 93, 94, 97, 98, 101, 91, 92, 95, 96, 99, 100, 102]:
        if m not in rows:
            continue
        x, y, w, h = coords[m]
        draw_team_match_box(svg, x, y, w, h, rows[m], lang)

    final_row = rows.get(104)
    if final_row:
        draw_team_match_box(svg, *final_box, final_row, lang, final=True)

    third_place_row = rows.get(103)
    if third_place_row:
        draw_team_match_box(svg, *third_place_box, third_place_row, lang)

    # Champion box.
    champ = final_row.get("projected_winner", "") if final_row else ""
    champ_prob_label = champion_probability_label(data, sid, champ, lang)
    svg.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="14" fill="#f4c542" stroke="{line}" stroke-width="4"/>')
    svg.append(svg_flag(champ, cx + 20, cy + 18, 36, 24))
    svg.append(svg_text(cx + cw - 18, cy + 20, round_label("Champion"), 13, "900", "#785500", "end"))
    svg.append(svg_text(cx + cw / 2 + 20, cy + 40, team_label(champ, lang), 22 if lang == "ja" else 19, "900", "#1b1600", "middle"))
    if champ_prob_label:
        svg.append(svg_text(cx + cw / 2 + 20, cy + 57, champ_prob_label, 11, "900", "#4f3900", "middle"))

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


# ---------------------------------------------------------------------
# HTML and PNG conversion
# ---------------------------------------------------------------------

def read_svg_for_inline(svg_path: Path, title: str) -> str:
    """Return an inline SVG block for a self-contained HTML report.

    The SVG files are still written as separate artifacts, but embedding them
    avoids broken-image placeholders when a browser, notebook preview, or
    document system opens the report without resolving adjacent relative files.
    """
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
    except OSError:
        # Fallback for partially generated folders.  In a normal run this branch
        # should not be used because SVG files are written before the report.
        return (
            f'<img class="embedded-svg" src="{esc(svg_path.name)}" '
            f'alt="{esc(title)}">'
        )

    svg_text = re.sub(
        r"<svg\b",
        f'<svg class="embedded-svg" role="img" aria-label="{esc(title)}"',
        svg_text,
        count=1,
    )
    return svg_text


def figure_block(svg_path: Path, title: str) -> str:
    return f'<div class="figure">{read_svg_for_inline(svg_path, title)}</div>'


def write_html_report(path: Path, lang: str, sid: str, data: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    weighted = sorted(data.get("weighted", []), key=lambda r: -as_float(r.get("champion", 0.0), 0.0))[:12]
    maxp = max([as_float(r.get("champion", 0.0), 0.0) for r in weighted] or [1.0])

    bars = []
    for i, r in enumerate(weighted, 1):
        p = as_float(r.get("champion", 0.0), 0.0)
        w = 100 * p / maxp if maxp > 0 else 0
        team = r.get("team", "")
        bars.append(
            f"<div class='barrow'><span>{i}</span><b>{esc(team_label(team, lang))}</b>"
            f"<div class='barbg'><div class='bar' style='width:{w:.1f}%'></div></div><b>{pct(p)}</b></div>"
        )

    css = f"""
body {{
  margin:0; padding:28px; background:#dce5f7; color:#111;
  font-family:{FONT_FAMILY};
}}
h1 {{ margin:0 0 6px; font-size:34px; }}
h2 {{ margin:30px 0 12px; font-size:24px; }}
.sub {{ color:#526077; font-weight:700; }}
.figure {{
  width:100%; max-width:1800px; display:block; margin:12px 0 28px;
  border:3px solid #111; background:#dce5f7; overflow:auto;
}}
.figure .embedded-svg {{
  width:100%; height:auto; display:block;
}}
.panel {{
  max-width:900px; background:#f6f8ff; border:3px solid #111;
  padding:14px 18px; margin:14px 0 28px;
}}
.barrow {{
  display:grid; grid-template-columns:34px 220px 1fr 70px;
  gap:10px; align-items:center; margin:7px 0;
}}
.barbg {{ height:16px; background:#cad5ea; border:1px solid #111; }}
.bar {{ height:100%; background:#2fbf71; }}
a {{ color:#0b4f9c; font-weight:800; }}
"""
    html_text = f"""<!doctype html>
<html lang="{esc(lang)}">
<head>
<meta charset="utf-8">
<title>{esc(tr(lang, 'html_title'))}</title>
<style>{css}</style>
</head>
<body>
<h1>{esc(tr(lang, 'title'))}</h1>
<div class="sub">{esc(tr(lang, 'subtitle'))} / {esc(tr(lang, 'scenario'))}: {esc(sid)}</div>

<h2>{esc(tr(lang, 'weighted_champion_prob'))}</h2>
<div class="sub">{esc(tr(lang, 'probability_note'))}</div>
<div class="panel">{''.join(bars)}</div>

<h2>{esc(tr(lang, 'group_title'))}</h2>
{figure_block(path.parent / f"group_stage_compact_{lang}.svg", tr(lang, 'group_title'))}

<h2>{esc(tr(lang, 'r32_title'))}</h2>
{figure_block(path.parent / f"r32_compact_{lang}.svg", tr(lang, 'r32_title'))}

<h2>{esc(tr(lang, 'bracket_title'))}</h2>
{figure_block(path.parent / f"knockout_r16_compact_{lang}.svg", tr(lang, 'bracket_title'))}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def convert_png(svg_path: Path, output_width: int = 1800) -> None:
    try:
        import cairosvg  # type: ignore
    except Exception:
        print("cairosvg is not installed; PNG conversion skipped.", file=sys.stderr)
        return
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(svg_path.with_suffix(".png")), output_width=output_width)
    except Exception as e:
        print(f"PNG conversion failed for {svg_path}: {e}", file=sys.stderr)


def write_index(path: Path, languages: Sequence[str]) -> None:
    items = []
    for lang in languages:
        label = tr(lang, "open_ja") if lang == "ja" else tr(lang, "open_en")
        items.append(f'<li><a href="{lang}/compact_report_{lang}.html">{esc(label)}</a></li>')
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Compact World Cup visualization</title>
<style>
body {{ font-family:{FONT_FAMILY}; background:#dce5f7; color:#111; padding:30px; }}
a {{ font-size:22px; font-weight:900; color:#0b4f9c; }}
li {{ margin:14px 0; }}
</style>
</head>
<body>
<h1>Compact World Cup visualization</h1>
<ul>{''.join(items)}</ul>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create compact bilingual World Cup two-stage visualizations.")
    ap.add_argument("--input-dir", type=Path, required=True, help="two_stage_worldcup_2026.py output directory")
    ap.add_argument("--output-dir", type=Path, default=None, help="output directory; default: INPUT_DIR/visuals_compact")
    ap.add_argument("--language", choices=("en", "ja", "both"), default="both")
    ap.add_argument("--scenario-limit", type=int, default=1, help="currently the first selected scenario is used")
    ap.add_argument("--no-png", action="store_true", help="skip PNG generation")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir or (args.input_dir / "visuals_compact")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_outputs(args.input_dir)
    if not data["selected"]:
        raise RuntimeError("selected_group_scenarios.csv is empty")

    # The compact bracket is intended as a single explainer view.  Use the first
    # selected scenario by default.
    sid = data["selected"][0]["scenario_id"]

    languages = ["en", "ja"] if args.language == "both" else [args.language]

    for lang in languages:
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        group_svg = lang_dir / f"group_stage_compact_{lang}.svg"
        r32_svg = lang_dir / f"r32_compact_{lang}.svg"
        ko_svg = lang_dir / f"knockout_r16_compact_{lang}.svg"

        write_group_stage_svg(group_svg, data, sid, lang)
        write_r32_svg(r32_svg, data, sid, lang)
        write_r16_bracket_svg(ko_svg, data, sid, lang)
        write_html_report(lang_dir / f"compact_report_{lang}.html", lang, sid, data)

        if not args.no_png:
            convert_png(group_svg)
            convert_png(r32_svg)
            convert_png(ko_svg)

    write_index(out_dir / "index.html", languages)
    print(f"Wrote compact visuals to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
