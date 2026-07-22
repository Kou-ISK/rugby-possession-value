from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SOURCE_COMMIT = "ee1e3b1cb98b879dac66372f19c886e2c69062b1"
SOURCE_BLOB = "6cc793ef8d21b5ec08dc7a21a37533ad4e7d6e35"
SOURCE_URL = (
    "https://raw.githubusercontent.com/WhartonSABI/rugby-ep/"
    f"{SOURCE_COMMIT}/data/phase_2018-19.csv"
)
EXPECTED_ROWS = 35_199

LOCATIONS = [
    "Goal-5m (own)",
    "5m-22m (own)",
    "22m-10m (own)",
    "10m-Half (own)",
    "Half-10m (opp)",
    "10m-22m (opp)",
    "22m-5m (opp)",
    "5m-Goal (opp)",
]
ORIGINS = [
    "Lineout",
    "Scrum",
    "Kick (Open Play)",
    "Turnover Steal",
    "Restart Kick",
    "Quick Tap",
]
SIDES = ["Left", "Centre", "Right"]
PHASE_BUCKETS = ["1", "2-3", "4-6", "7+"]

LOCATION_JA = {
    "Goal-5m (own)": "自陣G前5m",
    "5m-22m (own)": "自陣5-22m",
    "22m-10m (own)": "自陣22m-10m",
    "10m-Half (own)": "自陣10m-ハーフ",
    "Half-10m (opp)": "敵陣ハーフ-10m",
    "10m-22m (opp)": "敵陣10-22m",
    "22m-5m (opp)": "敵陣22m-5m",
    "5m-Goal (opp)": "敵陣G前5m",
}
ORIGIN_JA = {
    "Lineout": "ラインアウト",
    "Scrum": "スクラム",
    "Kick (Open Play)": "キックレシーブ",
    "Turnover Steal": "ターンオーバー",
    "Restart Kick": "リスタート",
    "Quick Tap": "クイックタップ",
}
OUTCOME_REWARD = {
    "For Converted Try (+7)": 7.0,
    "For Try (+5)": 5.0,
    "For Penalty Kick (+3)": 3.0,
    "For Drop Goal (+3)": 3.0,
    "Against Converted Try (-7)": -7.0,
    "Against Try (-5)": -5.0,
    "Against Penalty Kick (-3)": -3.0,
    "Against Drop Goal (-3)": -3.0,
    "End of Half / Match (0)": 0.0,
}
OUTCOMES = list(OUTCOME_REWARD)


def phase_bucket(phase: int) -> str:
    if phase <= 1:
        return "1"
    if phase <= 3:
        return "2-3"
    if phase <= 6:
        return "4-6"
    return "7+"


def git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def download_and_validate(path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        content = response.read()
    actual_blob = git_blob_sha(content)
    if actual_blob != SOURCE_BLOB:
        raise RuntimeError(f"Dataset blob mismatch: {actual_blob}")
    path.write_bytes(content)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    return frame


def normalise(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total > 0:
        return values / total
    return np.full(values.shape, 1.0 / len(values), dtype=float)


def build_markov_model(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["match_id"] = (
        data["Round"].astype(str)
        + "|"
        + data["Home"].astype(str)
        + "|"
        + data["Away"].astype(str)
    )
    data = data.sort_values(["match_id", "ID"], kind="stable").reset_index(drop=True)
    data["phase_bucket"] = data["Phase"].astype(int).map(phase_bucket)

    states = [
        (origin, location, side, bucket)
        for origin in ORIGINS
        for location in LOCATIONS
        for side in SIDES
        for bucket in PHASE_BUCKETS
    ]
    state_index = {state: index for index, state in enumerate(states)}
    outcome_index = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    n_states = len(states)
    n_outcomes = len(OUTCOMES)

    same_counts = np.zeros((n_states, n_states), dtype=float)
    opp_counts = np.zeros((n_states, n_states), dtype=float)
    absorb_counts = np.zeros((n_states, n_outcomes), dtype=float)
    visits = np.zeros(n_states, dtype=int)

    for index, row in data.iterrows():
        source = (
            str(row["Play_Start"]),
            str(row["Location"]),
            str(row["Side"]),
            str(row["phase_bucket"]),
        )
        source_i = state_index.get(source)
        if source_i is None:
            continue
        visits[source_i] += 1
        at_match_end = (
            index == len(data) - 1
            or data.loc[index + 1, "match_id"] != row["match_id"]
        )
        if at_match_end:
            absorb_counts[source_i, outcome_index[str(row["Outcome"])]] += 1
            continue

        nxt = data.loc[index + 1]
        flipped = str(nxt["Team_In_Poss"]) != str(row["Team_In_Poss"])
        next_difference = float(nxt["Points_Difference"])
        if flipped:
            next_difference = -next_difference
        score_delta = next_difference - float(row["Points_Difference"])

        if abs(score_delta) > 1e-9:
            absorb_counts[source_i, outcome_index[str(row["Outcome"])]] += 1
            continue

        target = (
            str(nxt["Play_Start"]),
            str(nxt["Location"]),
            str(nxt["Side"]),
            str(nxt["phase_bucket"]),
        )
        target_i = state_index.get(target)
        if target_i is None:
            continue
        if flipped:
            opp_counts[source_i, target_i] += 1
        else:
            same_counts[source_i, target_i] += 1

    global_same_raw = same_counts.sum(axis=0)
    global_opp_raw = opp_counts.sum(axis=0)
    global_abs_raw = absorb_counts.sum(axis=0)
    global_split_raw = np.array(
        [same_counts.sum(), opp_counts.sum(), absorb_counts.sum()], dtype=float
    )

    p_same = np.zeros_like(same_counts)
    p_opp = np.zeros_like(opp_counts)
    p_abs = np.zeros_like(absorb_counts)

    alpha_split = 12.0
    alpha_transition = 18.0
    alpha_absorb = 12.0

    for i, state in enumerate(states):
        origin, location, _, bucket = state
        location_peers = [
            j for j, peer in enumerate(states) if peer[1] == location and peer[3] == bucket
        ]
        origin_peers = [
            j for j, peer in enumerate(states) if peer[0] == origin and peer[3] == bucket
        ]

        same_prior = normalise(
            0.55 * same_counts[location_peers].sum(axis=0)
            + 0.30 * same_counts[origin_peers].sum(axis=0)
            + 0.15 * global_same_raw
        )
        opp_prior = normalise(
            0.55 * opp_counts[location_peers].sum(axis=0)
            + 0.30 * opp_counts[origin_peers].sum(axis=0)
            + 0.15 * global_opp_raw
        )
        absorb_prior = normalise(
            0.55 * absorb_counts[location_peers].sum(axis=0)
            + 0.30 * absorb_counts[origin_peers].sum(axis=0)
            + 0.15 * global_abs_raw
        )

        split_prior = normalise(
            0.55
            * np.array(
                [
                    same_counts[location_peers].sum(),
                    opp_counts[location_peers].sum(),
                    absorb_counts[location_peers].sum(),
                ]
            )
            + 0.30
            * np.array(
                [
                    same_counts[origin_peers].sum(),
                    opp_counts[origin_peers].sum(),
                    absorb_counts[origin_peers].sum(),
                ]
            )
            + 0.15 * global_split_raw
        )

        same_n = same_counts[i].sum()
        opp_n = opp_counts[i].sum()
        absorb_n = absorb_counts[i].sum()
        total_n = same_n + opp_n + absorb_n

        split = (
            np.array([same_n, opp_n, absorb_n]) + alpha_split * split_prior
        ) / (total_n + alpha_split)
        same_cond = (
            same_counts[i] + alpha_transition * same_prior
        ) / (same_n + alpha_transition)
        opp_cond = (
            opp_counts[i] + alpha_transition * opp_prior
        ) / (opp_n + alpha_transition)
        absorb_cond = (
            absorb_counts[i] + alpha_absorb * absorb_prior
        ) / (absorb_n + alpha_absorb)

        p_same[i] = split[0] * same_cond
        p_opp[i] = split[1] * opp_cond
        p_abs[i] = split[2] * absorb_cond

    rewards = np.array([OUTCOME_REWARD[o] for o in OUTCOMES], dtype=float)
    system = np.eye(n_states) - p_same + p_opp
    values = np.linalg.solve(system, p_abs @ rewards)

    state_rows = []
    for i, state in enumerate(states):
        origin, location, side, bucket = state
        state_rows.append(
            {
                "origin": origin,
                "location": location,
                "side": side,
                "phase_bucket": bucket,
                "epv": float(values[i]),
                "observations": int(visits[i]),
            }
        )
    state_frame = pd.DataFrame(state_rows)

    heat_rows = []
    for location in LOCATIONS:
        for origin in ORIGINS:
            group = state_frame[
                (state_frame["location"] == location)
                & (state_frame["origin"] == origin)
                & (state_frame["phase_bucket"] == "1")
            ]
            weights = group["observations"].to_numpy(dtype=float) + 2.0
            heat_rows.append(
                {
                    "エリア": LOCATION_JA[location],
                    "ポゼッション起点": ORIGIN_JA[origin],
                    "EPV": float(np.average(group["epv"], weights=weights)),
                    "観測数": int(group["observations"].sum()),
                }
            )
    return state_frame, pd.DataFrame(heat_rows)


def draw_heatmap(heat_frame: pd.DataFrame, output_path: Path) -> None:
    epv = heat_frame.pivot(index="エリア", columns="ポゼッション起点", values="EPV")
    counts = heat_frame.pivot(index="エリア", columns="ポゼッション起点", values="観測数")
    row_order = [LOCATION_JA[location] for location in LOCATIONS]
    col_order = [ORIGIN_JA[origin] for origin in ORIGINS]
    epv = epv.reindex(index=row_order, columns=col_order)
    counts = counts.reindex(index=row_order, columns=col_order)

    # Colour-blind-friendlier diverging scale: orange = negative, neutral = zero,
    # blue = positive. The scale is symmetric around zero to avoid exaggeration.
    palette = LinearSegmentedColormap.from_list(
        "epv_diverging", ["#B35806", "#F7F7F7", "#2166AC"], N=256
    )
    max_abs = float(np.nanmax(np.abs(epv.to_numpy())))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(15.5, 9.2), dpi=220)
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_facecolor("#F5F7FA")
    image = ax.imshow(epv.to_numpy(), cmap=palette, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(epv.columns)))
    ax.set_xticklabels(epv.columns, fontsize=11.5, fontweight="bold")
    ax.set_yticks(np.arange(len(epv.index)))
    ax.set_yticklabels(epv.index, fontsize=11.5, fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=10)

    ax.set_xticks(np.arange(-0.5, len(epv.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(epv.index), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_i in range(len(epv.index)):
        for col_i in range(len(epv.columns)):
            value = float(epv.iloc[row_i, col_i])
            count = int(counts.iloc[row_i, col_i])
            rgba = palette(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_colour = "#111827" if luminance > 0.62 else "#FFFFFF"
            ax.text(
                col_i,
                row_i - 0.08,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=text_colour,
            )
            ax.text(
                col_i,
                row_i + 0.25,
                f"n={count:,}",
                ha="center",
                va="center",
                fontsize=8.7,
                color=text_colour,
                alpha=0.9,
            )

    ax.set_title(
        "エリア × ポゼッション起点　得点期待値（EPV）",
        fontsize=23,
        fontweight="bold",
        color="#0F2747",
        pad=58,
        loc="left",
    )
    ax.text(
        0,
        1.105,
        "保持交代後も次の得点イベントまで追う、符号付きマルコフ連鎖モデル",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#475569",
        ha="left",
    )

    colourbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    colourbar.set_label("EPV（0を基準）", fontsize=11)
    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(labelsize=9)

    fig.text(
        0.06,
        0.035,
        "オレンジ＝相手の次得点が優勢 / 白＝中立 / 青＝自チームの次得点が優勢。色域は0を中心に左右対称。",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.06,
        0.017,
        "各セルはモデル推定EPV。nは該当エリア×起点×第1フェーズの実観測数で、推定の根拠量を示す。",
        fontsize=9.5,
        color="#64748B",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.04, 0.065, 0.98, 0.91])
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    raw_path = Path("data/raw/phase_2018-19.csv")
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = download_and_validate(raw_path)
    state_frame, heat_frame = build_markov_model(frame)
    state_frame.to_csv(output_dir / "epv_state_values_real.csv", index=False)
    heat_frame.to_csv(output_dir / "epv_heatmap_real.csv", index=False, encoding="utf-8-sig")
    draw_heatmap(heat_frame, output_dir / "epv_heatmap_real.png")

    metadata = {
        "source_repository": "WhartonSABI/rugby-ep",
        "source_commit": SOURCE_COMMIT,
        "source_blob": SOURCE_BLOB,
        "source_url": SOURCE_URL,
        "rows": int(len(frame)),
        "matches": int(frame[["Round", "Home", "Away"]].drop_duplicates().shape[0]),
        "model": "signed cross-possession Markov reward process with hierarchical empirical-Bayes smoothing",
        "colour_scale": "symmetric diverging scale centred at EPV=0; orange negative, white zero, blue positive",
    }
    (output_dir / "epv_heatmap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
