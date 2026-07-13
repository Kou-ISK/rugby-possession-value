from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import CrossPossessionEPV

LOCATION_JA = {
    "Goal-5m (own)": "自陣ゴール前5m",
    "5m-22m (own)": "自陣5-22m",
    "22m-10m (own)": "自陣22m-10m",
    "10m-Half (own)": "自陣10m-ハーフ",
    "Half-10m (opp)": "敵陣ハーフ-10m",
    "10m-22m (opp)": "敵陣10-22m",
    "22m-5m (opp)": "敵陣22-5m",
    "5m-Goal (opp)": "敵陣ゴール前5m",
}
ORIGIN_JA = {
    "Lineout": "ラインアウト",
    "Scrum": "スクラム",
    "Kick (Open Play)": "キックレシーブ起点",
    "Turnover Steal": "ターンオーバー",
    "Restart Kick": "リスタート",
    "Quick Tap": "クイックタップ",
}
SIDE_JA = {"Left": "左", "Centre": "中央", "Right": "右"}


def write_tables(
    model: CrossPossessionEPV,
    output: str | Path,
    intervals: pd.DataFrame | None = None,
) -> None:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    states = model.state_frame()
    if intervals is not None:
        states = states.merge(intervals, on="state_key", how="left")
    states.to_csv(target / "state_values.csv", index=False)

    starts = states[states["phase_bucket"] == "1"].copy()
    starts.sort_values(["x", "origin", "side"]).to_csv(
        target / "start_state_values.csv", index=False
    )

    student = starts[
        [
            "location",
            "side",
            "origin",
            "epv",
            "p_score_for",
            "p_try_for",
            "p_score_against",
            "p_no_score",
            "state_visits",
            *(["epv_p05", "epv_p95"] if "epv_p05" in starts else []),
        ]
    ].copy()
    student.insert(0, "エリア", student.pop("location").map(LOCATION_JA))
    student.insert(1, "横位置", student.pop("side").map(SIDE_JA))
    student.insert(
        2,
        "ポゼッション起点",
        student.pop("origin").map(ORIGIN_JA).fillna(starts["origin"]),
    )
    student = student.rename(
        columns={
            "epv": "得点期待値_EPV",
            "p_score_for": "自チーム得点確率",
            "p_try_for": "自チームトライ確率",
            "p_score_against": "相手得点確率",
            "p_no_score": "無得点終了確率",
            "state_visits": "標本数",
            "epv_p05": "EPV_下限90pct",
            "epv_p95": "EPV_上限90pct",
        }
    )
    student.to_csv(target / "student_table.csv", index=False)
    model.transition_frame(minimum_probability=0.001).to_csv(
        target / "transition_probabilities.csv", index=False
    )
