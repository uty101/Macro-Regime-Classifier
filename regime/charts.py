"""The output charts. features_review_chart is built in step 2.4; the three section 7 charts in step 7.1.

Every ``savefig`` passes ``metadata={"Software": None}`` so the PNG carries no
matplotlib version string and two runs produce byte-identical files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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


STATE_COLOURS = ("#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb", "#fcbba1")
CPI_COLOUR = "#b2432f"


def _last_refit_rows(param_drift: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``param_drift`` belonging to its last refit date.

    ``sort_values`` then ``iloc[-1]`` rather than ``groupby().last()``: the
    latter picks a group's last *row* under whatever order the frame happened
    to arrive in, which is not the same question.
    """
    dates = pd.to_datetime(param_drift["refit_date"])
    last = dates.sort_values().iloc[-1]
    return param_drift.loc[dates.to_numpy() == last]


def state_legend_labels(param_drift: pd.DataFrame, cfg: Config) -> dict[int, str]:
    """``{k: "state k: mean dgs10_chg12 = x, mean cpi_3m_ann = y"}`` from the last refit.

    The two features are the anchor feature and the anchor tiebreak feature
    (``cfg.hmm_anchor_feature``, ``cfg.hmm_anchor_tiebreak_feature``), so the
    legend says what the state numbering was ordered on. Two decimals, in
    z-units, because ``param_drift.csv`` is written from the standardised
    model input.
    """
    rows = _last_refit_rows(param_drift)
    anchor, tiebreak = cfg.hmm_anchor_feature, cfg.hmm_anchor_tiebreak_feature
    labels = {}
    for state in sorted(rows["state"].unique()):
        of_state = rows.loc[rows["state"] == state].set_index("feature")["mean"]
        labels[int(state)] = (
            f"state {int(state)}: mean {anchor} = {of_state[anchor]:.2f}, "
            f"mean {tiebreak} = {of_state[tiebreak]:.2f}"
        )
    return labels


def _shade_by_label(ax, labels: pd.DataFrame, start, end) -> None:
    """Background shading of ``ax`` by the hard label, one span per constant run."""
    frame = labels.loc[(labels.index >= start) & (labels.index <= end), "label"].dropna()
    if frame.empty:
        return
    values = frame.to_numpy()
    dates = frame.index
    run_start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[run_start]:
            left = dates[run_start]
            right = dates[i] if i < len(values) else end
            ax.axvspan(left, right, color=STATE_COLOURS[int(values[run_start]) % len(STATE_COLOURS)], zorder=0)
            run_start = i


def regimes_timeline(
    panel_raw: pd.DataFrame,
    filt_labels: pd.DataFrame,
    smooth_labels: pd.DataFrame,
    param_drift: pd.DataFrame,
    path: str,
    cfg: Config,
) -> None:
    """Two panels over ``features_from`` to ``sample_end``, shaded by regime.

    Top panel: ``dgs10_level`` and ``cpi_3m_ann`` on twin axes, background
    shaded by the **filtered** hard label. Bottom panel: the identical series
    shaded by the **smoothed** label. The two panels are the picture of
    question 4 — one history labelled with and without hindsight — and the
    visible difference between them is what ``filtered_smoothed_gap.csv``
    measures.

    Legend entries come from the last refit's ``param_drift.csv`` rows.
    """
    start, end = pd.Timestamp(cfg.sample_features_from), pd.Timestamp(cfg.sample_end)
    frame = panel_raw.loc[(panel_raw.index >= start) & (panel_raw.index <= end)]
    legend = state_legend_labels(param_drift, cfg)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, labels, title in (
        (axes[0], filt_labels, "filtered labels (real time)"),
        (axes[1], smooth_labels, "smoothed labels (full-sample hindsight)"),
    ):
        _shade_by_label(ax, labels, start, end)
        ax.plot(frame.index, frame["dgs10_level"], color=LINE_COLOUR, linewidth=1.1)
        ax.set_ylabel("dgs10_level", fontsize=9, color=LINE_COLOUR)
        twin = ax.twinx()
        twin.plot(frame.index, frame["cpi_3m_ann"], color=CPI_COLOUR, linewidth=1.1)
        twin.set_ylabel("cpi_3m_ann", fontsize=9, color=CPI_COLOUR)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(start, end)
        ax.spines[["top"]].set_visible(False)
        ax.tick_params(labelsize=8)
        twin.tick_params(labelsize=8)

    handles = [
        matplotlib.patches.Patch(color=STATE_COLOURS[k % len(STATE_COLOURS)], label=legend[k])
        for k in sorted(legend)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8, frameon=False)
    fig.suptitle(f"Regimes, {start.date()} to {end.date()}", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.outputs_chart_dpi, metadata={"Software": None})
    plt.close(fig)


def conditional_sharpe_heatmap(stats: pd.DataFrame, path: str, cfg: Config):
    """Factor x state heatmap of the conditional Sharpe, each cell ``"SR / n"``.

    A **black border** marks a cell whose bootstrap interval excludes zero.
    That border says the factor pays in that state; it does not say it pays
    *differently* there, which is the pairwise-difference table's question and
    not this chart's. Returns the ``Axes`` so a test can count the borders.
    """
    factors = [f for f in cfg.strategy_factors if f in set(stats["factor"])]
    states = sorted(int(k) for k in stats["state"].unique())
    grid = stats.pivot(index="factor", columns="state", values="sharpe").reindex(index=factors, columns=states)
    counts = stats.pivot(index="factor", columns="state", values="n").reindex(index=factors, columns=states)

    fig, ax = plt.subplots(figsize=(1.9 * len(states) + 3.0, 0.7 * len(factors) + 2.0))
    limit = float(np.nanmax(np.abs(grid.to_numpy(dtype="float64")))) if len(grid) else 1.0
    image = ax.imshow(grid.to_numpy(dtype="float64"), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(states)), [f"state {k}" for k in states], fontsize=9)
    ax.set_yticks(range(len(factors)), factors, fontsize=9)
    for i in range(len(factors)):
        for j in range(len(states)):
            ax.text(
                j, i, f"{grid.iat[i, j]:.2f} / {int(counts.iat[i, j])}",
                ha="center", va="center", fontsize=8, color="#1a1a1a",
            )
    for row in stats.loc[stats["excludes_zero"].astype(bool)].itertuples(index=False):
        if row.factor in factors:
            ax.add_patch(
                plt.Rectangle(
                    (states.index(int(row.state)) - 0.5, factors.index(row.factor) - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=2.0,
                )
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="annualised Sharpe")
    ax.set_title("conditional Sharpe by regime — black border: interval excludes 0", fontsize=10, loc="left")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.outputs_chart_dpi, metadata={"Software": None})
    plt.close(fig)
    return ax


def timed_vs_static(backtests: dict[str, pd.DataFrame], path: str, cfg: Config) -> None:
    """Cumulative net return of static, timed-filtered and timed-smoothed on one log axis.

    The headline cell's (eta, lag, cost) and nothing else. A log axis because
    three cumulative wealth curves that stay within a few per cent of each
    other for twenty years are unreadable on a linear one — and that they do
    is the point of the chart.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colours = {"static": "#6b6b6b", "timed_filtered": LINE_COLOUR, "timed_smoothed": CPI_COLOUR}
    for name, frame in backtests.items():
        wealth = (1.0 + frame["net_ret"].astype("float64")).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), color=colours.get(name), linewidth=1.2, label=name)
    ax.set_yscale("log")
    ax.grid(True, color=GRID_COLOUR, linewidth=0.5, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylabel("cumulative net return (log scale)", fontsize=9)
    ax.set_title(
        f"timed vs static — eta {cfg.strategy_headline_eta}, lag {cfg.strategy_headline_lag}, "
        f"{cfg.strategy_headline_cost_bp} bp",
        fontsize=10, loc="left",
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.outputs_chart_dpi, metadata={"Software": None})
    plt.close(fig)
