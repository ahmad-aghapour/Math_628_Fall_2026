"""CUSUM sampling and triple-barrier labeling utilities."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, span: int = 30) -> pd.Series:
    """Exponentially weighted volatility of one-period log returns."""
    return np.log(close.astype(float)).diff().ewm(span=span, min_periods=span).std()


def cusum_filter(log_price: pd.Series, threshold: pd.Series) -> pd.DatetimeIndex:
    """Return timestamps where a symmetric CUSUM process crosses its threshold."""
    changes, limits = log_price.astype(float).diff().align(threshold.astype(float), join="inner")
    positive_sum = 0.0
    negative_sum = 0.0
    events = []

    for timestamp, change in changes.dropna().items():
        limit = limits.loc[timestamp]
        if pd.isna(limit) or limit <= 0:
            continue

        positive_sum = max(0.0, positive_sum + change)
        negative_sum = min(0.0, negative_sum + change)

        if positive_sum > limit:
            positive_sum = 0.0
            events.append(timestamp)
        elif negative_sum < -limit:
            negative_sum = 0.0
            events.append(timestamp)

    return pd.DatetimeIndex(events).drop_duplicates()


def add_vertical_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    holding_days: int = 20,
) -> pd.Series:
    """Map each event to the first observation on or after its time barrier."""
    positions = close.index.searchsorted(events + pd.Timedelta(days=holding_days))
    valid = positions < len(close)
    return pd.Series(close.index[positions[valid]], index=events[valid], name="vertical_time")


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    target: pd.Series,
    vertical_times: pd.Series,
    pt: float = 1.0,
    sl: float = 1.0,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Label each event by the first volatility-scaled barrier touched."""
    rows = []
    usable = events.intersection(vertical_times.index).intersection(target.dropna().index)
    if side is None:
        side = pd.Series(1.0, index=usable)

    for start in usable:
        vertical_time = vertical_times.loc[start]
        width = float(target.loc[start])
        trade_side = float(side.reindex([start]).iloc[0])
        if not np.isfinite(width) or width <= 0 or not np.isfinite(trade_side):
            continue

        raw_path = close.loc[start:vertical_time] / close.loc[start] - 1.0
        signed_path = trade_side * raw_path
        upper_hits = signed_path[signed_path >= pt * width]
        lower_hits = signed_path[signed_path <= -sl * width]

        candidates = [(vertical_time, 0, "vertical", 1)]
        if len(upper_hits):
            candidates.append((upper_hits.index[0], 1, "profit_taking", 0))
        if len(lower_hits):
            candidates.append((lower_hits.index[0], -1, "stop_loss", 0))

        exit_time, label, barrier, _ = min(candidates, key=lambda item: (item[0], item[3]))
        raw_return = close.loc[exit_time] / close.loc[start] - 1.0
        rows.append(
            {
                "event_time": start,
                "exit_time": exit_time,
                "vertical_time": vertical_time,
                "target": width,
                "barrier": barrier,
                "label": label,
                "raw_return": raw_return,
                "strategy_return": trade_side * raw_return,
                "holding_days": (exit_time - start).days,
            }
        )

    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="event_time"))
    return pd.DataFrame(rows).set_index("event_time").sort_index()


# Compatibility names used by the existing Week 5 notebook.
def daily_volatility_with_log_returns(close: pd.Series, span: int = 100) -> pd.Series:
    return daily_volatility(close, span)


def cusum_filter_events_dynamic_threshold(
    prices: pd.Series, threshold: pd.Series
) -> pd.DatetimeIndex:
    return cusum_filter(prices, threshold)


def vertical_barrier(
    close: pd.Series, time_events: pd.DatetimeIndex, number_days: int
) -> pd.Series:
    return add_vertical_barrier(close, time_events, number_days)


def meta_events(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    ptsl: List[float],
    target: pd.Series,
    return_min: float,
    num_threads: int,
    timestamp: pd.Series = False,
    side: Optional[pd.Series] = None,
):
    """Compatibility wrapper returning the columns expected by Week 5."""
    del num_threads
    filtered_target = target.reindex(time_events).dropna()
    filtered_target = filtered_target[filtered_target > return_min]
    selected_events = filtered_target.index

    if timestamp is False:
        vertical_times = pd.Series(close.index[-1], index=selected_events)
    else:
        vertical_times = timestamp.reindex(selected_events).dropna()
        selected_events = selected_events.intersection(vertical_times.index)

    pt = float(ptsl[0])
    sl = float(ptsl[1] if len(ptsl) > 1 else ptsl[0])
    event_side = None if side is None else side.reindex(selected_events)
    labeled = triple_barrier_labels(
        close,
        selected_events,
        filtered_target,
        vertical_times,
        pt=pt,
        sl=sl,
        side=event_side,
    )

    events = pd.DataFrame(index=labeled.index)
    events["End Time"] = labeled["exit_time"]
    events["Base Width"] = labeled["target"]
    events["target"] = labeled["target"]
    events["timestamp"] = labeled["vertical_time"]
    events["Label"] = labeled["label"]
    events["Return of Label"] = labeled["strategy_return"]
    if side is not None:
        events["Side"] = event_side.reindex(events.index)

    touches = labeled[["exit_time", "barrier"]].rename(columns={"exit_time": "End Time"})
    return events, touches


def meta_labeling(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Return Week 5-compatible label and return columns."""
    if {"Label", "Return of Label"}.issubset(events.columns):
        out = events[["End Time", "Return of Label", "Label"]].copy()
        if "Side" in events:
            out["Label"] = (out["Label"] == 1).astype(int)
            out["Side"] = events["Side"]
        return out

    valid = events.dropna(subset=["End Time"])
    out = pd.DataFrame(index=valid.index)
    out["End Time"] = valid["End Time"]
    start_price = close.reindex(valid.index).to_numpy()
    end_price = close.reindex(valid["End Time"].to_numpy()).to_numpy()
    out["Return of Label"] = end_price / start_price - 1.0
    out["Label"] = np.sign(out["Return of Label"]).astype(int)
    return out
