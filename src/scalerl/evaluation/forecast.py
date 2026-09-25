"""Score demand forecasts against the traffic that actually occurred.

Scoring happens only after an episode: each forecast is joined by its
``target_tick`` to the completed step with that ``info["tick"]``. Forecasts
whose target tick never occurred (beyond the episode end) are ignored.
Actual values never flow back to the controller that made the forecast.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class Forecast(Protocol):
    @property
    def target_tick(self) -> int: ...

    @property
    def forecast_rps(self) -> float: ...


@dataclass(frozen=True)
class ForecastAccuracy:
    """Error of matured forecasts, in requests per second.

    MAPE is deliberately omitted: demand can be near zero.
    """

    forecast_count: int
    forecast_mae_rps: float | None
    forecast_rmse_rps: float | None
    forecast_bias_rps: float | None

    def as_metrics(self) -> dict[str, float]:
        """Metrics to log; undefined errors (no scored forecasts) are omitted."""
        return {name: float(value) for name, value in vars(self).items() if value is not None}


def score_forecasts(
    forecasts: Iterable[Forecast], infos: Sequence[Mapping[str, Any]]
) -> ForecastAccuracy:
    """Compare each forecast with the request rate of the tick it targeted."""
    actual = {int(info["tick"]): float(info["request_rate"]) for info in infos}
    errors = [
        forecast.forecast_rps - actual[forecast.target_tick]
        for forecast in forecasts
        if forecast.target_tick in actual
    ]
    if not errors:
        return ForecastAccuracy(0, None, None, None)
    return ForecastAccuracy(
        forecast_count=len(errors),
        forecast_mae_rps=statistics.fmean(abs(error) for error in errors),
        forecast_rmse_rps=math.sqrt(statistics.fmean(error * error for error in errors)),
        forecast_bias_rps=statistics.fmean(errors),
    )
