"""The output charts. features_review_chart is built in step 2.4; the three section 7 charts in step 7.1.

Every ``savefig`` passes ``metadata={"Software": None}`` so the PNG carries no
matplotlib version string and two runs produce byte-identical files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from regime.config import Config  # noqa: E402

LINE_COLOUR = "#2f6db5"
GRID_COLOUR = "#d9d9d9"


def features_review_chart(raw: pd.DataFrame, path: str, cfg: Config) -> None:
    """A 5 x 2 grid of the ten raw features from ``sample_features_from`` to ``sample_end`` as a PNG.

    One panel per column of ``cfg.features_core`` then ``cfg.features_robustness``,
    in that order, each titled with the column name; a dashed line marks
    ``sample_first_window_end``. Written at ``cfg.outputs_chart_dpi``.
    """
    features = list(cfg.features_core) + list(cfg.features_robustness)
    start, end = pd.Timestamp(cfg.sample_features_from), pd.Timestamp(cfg.sample_end)
    window_end = pd.Timestamp(cfg.sample_first_window_end)
    frame = raw.loc[(raw.index >= start) & (raw.index <= end), features]

    fig, axes = plt.subplots(5, 2, figsize=(12, 13), sharex=True)
    for ax, feature in zip(axes.ravel(), features):
        ax.plot(frame.index, frame[feature], color=LINE_COLOUR, linewidth=1.0)
        if frame[feature].min() < 0 < frame[feature].max():
            ax.axhline(0, color=GRID_COLOUR, linewidth=0.8)
        ax.axvline(window_end, color="#8c8c8c", linewidth=0.8, linestyle="--")
        ax.set_title(feature, fontsize=10, loc="left")
        ax.grid(True, color=GRID_COLOUR, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlim(start, end)
    fig.suptitle(
        f"Raw features, {start.date()} to {end.date()} (dashed: first window end {window_end.date()})",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.outputs_chart_dpi, metadata={"Software": None})
    plt.close(fig)
