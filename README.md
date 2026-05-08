# Adding a New Visualization to CHAP

A practical tutorial for adding a custom plotter to CHAP, verifying that it
works locally, and understanding how the same pattern extends to other
visualization types.

This guide accompanies the reference implementation at
[Isolated_visualisation](https://github.com/norajeanett/Isolated_visualisation),
which contains four working plotters that follow the contract described here.


## What you are building

A new plotter in CHAP is:

- A Python class subclassing `Plotter`
- Implementing a single `plot()` method
- Returning the path of a written PNG
- Registered in CHAP's `available_plotters` dictionary

Once registered, the plotter is available through the CLI, the REST API,
and the Modelling App without any frontend changes.


## Understand the data you receive

Your `plot()` method receives two flat pandas DataFrames.

**Observations** — one row per (location, time period):

| Column          | Description                            |
| --------------- | -------------------------------------- |
| `location`      | Location identifier                    |
| `time_period`   | ISO time key (e.g. `2024W01`)          |
| `disease_cases` | Observed value                         |

**Forecasts** — one row per (location, time period, horizon, sample):

| Column             | Description                                  |
| ------------------ | -------------------------------------------- |
| `location`         | Location identifier                          |
| `time_period`      | Forecasted period                            |
| `horizon_distance` | Steps ahead                                  |
| `sample`           | Sample index from the predictive distribution |
| `forecast`         | Predicted value                              |

Example shape:

```text
observations
   location  time_period  disease_cases
0  OrgUnit1  2024W01      120
1  OrgUnit1  2024W02      150
2  OrgUnit2  2024W01       80

forecasts
   location  time_period  horizon_distance  sample  forecast
0  OrgUnit1  2024W01      1                 0       118
1  OrgUnit1  2024W01      1                 1       123
2  OrgUnit1  2024W01      1                 2       121
3  OrgUnit1  2024W02      1                 0       155
4  OrgUnit1  2024W02      1                 1       149
```

Each (location, time_period, horizon_distance) tuple appears multiple times
— once per posterior sample. Most plotters collapse this dimension before
comparing forecasts to observations (e.g. via `groupby(...).mean()` or
`.quantile()`). This aggregation step is central to nearly every
visualization in CHAP.


## The plotter interface

Every plotter subclasses the abstract `Plotter` class:

```python
from abc import ABC, abstractmethod
import pandas as pd


class Plotter(ABC):
    name: str = "base"

    @abstractmethod
    def plot(
        self,
        forecasts: pd.DataFrame,
        observations: pd.DataFrame,
        out_path: str,
    ) -> str:
        ...
```

Three things to remember about the contract:

1. The argument order is `forecasts, observations, out_path` — forecasts
   first.
2. The method writes its output to `out_path` (PNG via
   `chart.save(out_path, scale_factor=2)`) and returns that same path.
3. The class-level `name` attribute is used as the default filename stem
   and as the registry key.


## Add your first plotter

We will add a Mean Absolute Error (MAE) by horizon plotter. It is a good
first example because it uses standard columns, requires only one
aggregation, and exercises every part of the contract.

### Create the file

Inside `chap_core/assessment/plotters/`, create:

```
mae_by_horizon_plot.py
```

### Implement the class

```python
import os
import pandas as pd
import altair as alt

from chap_core.assessment.plotters.base import Plotter


class MaeByHorizonPlotter(Plotter):
    """Mean absolute error as a function of forecast horizon."""

    name: str = "mae_by_horizon"

    def plot(
        self,
        forecasts: pd.DataFrame,
        observations: pd.DataFrame,
        out_path: str,
    ) -> str:
        # 1. Collapse posterior samples to a point forecast (mean)
        point_forecasts = (
            forecasts
            .groupby(["location", "time_period", "horizon_distance"])["forecast"]
            .mean()
            .reset_index()
            .rename(columns={"forecast": "prediction"})
        )

        # 2. Join predictions with observations
        merged = point_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]],
            on=["location", "time_period"],
            how="left",
        )

        # 3. Compute absolute error per row
        merged["abs_error"] = (
            merged["prediction"] - merged["disease_cases"]
        ).abs()

        # 4. Aggregate by horizon
        mae = (
            merged.groupby("horizon_distance")["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"abs_error": "mae"})
        )

        # 5. Build the Altair chart
        chart = (
            alt.Chart(mae)
            .mark_line(point=True)
            .encode(
                x=alt.X("horizon_distance:Q", title="Horizon distance"),
                y=alt.Y("mae:Q", title="Mean absolute error"),
                tooltip=["horizon_distance", "mae"],
            )
            .properties(
                width=500,
                height=300,
                title="Mean absolute error by horizon",
            )
        )

        # 6. Save and return
        outdir = os.path.dirname(out_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        chart.save(out_path, scale_factor=2)
        return out_path
```

This is the same five-step pattern that every plotter in the reference
implementation follows: filter (optional), collapse samples, merge,
compute, render.

### Register the plotter

CHAP discovers plotters through an `available_plotters` dictionary,
mirroring the pattern already used for metrics in
`chap_core/assessment/metrics/__init__.py`. Add an entry inside:

```
chap_core/assessment/plotters/__init__.py
```

For example:

```python
from chap_core.assessment.plotters.mae_by_horizon_plot import MaeByHorizonPlotter
# ...other imports...

available_plotters = {
    HeatmapPlotter.name:        HeatmapPlotter,
    ScatterPlotter.name:        ScatterPlotter,
    OutbreakProbPlotter.name:   OutbreakProbPlotter,
    MaeByHorizonPlotter.name:   MaeByHorizonPlotter,  # <-- new
}
```

If you forget this step, the plotter will not appear in the CLI or the
Modelling App.


## Test it

### Option A — Unit test against flat fixtures

```python
import pandas as pd
from chap_core.assessment.plotters.mae_by_horizon_plot import MaeByHorizonPlotter


def test_mae_by_horizon_plotter(tmp_path, flat_observations, flat_forecasts):
    out_path = tmp_path / "mae.png"
    plotter = MaeByHorizonPlotter()
    written = plotter.plot(
        pd.DataFrame(flat_forecasts),
        pd.DataFrame(flat_observations),
        str(out_path),
    )
    assert written == str(out_path)
    assert out_path.exists()
```

This verifies that the transformation runs end-to-end and that a PNG is
produced on disk.

### Option B — End-to-end through the CLI

```bash
chap eval my_model data.csv evaluation.nc
chap plot-backtest evaluation.nc mae.png --plot-type mae_by_horizon
```

If both commands succeed, the plotter is wired in correctly.


## Verify in the Modelling App

Once registered, the plotter is automatically available through:

```
GET /visualization/plotters/
GET /visualization/plotters/{plot_name}/{backtest_id}
```

No frontend changes are required.


## How to think about plotters in CHAP

Every plotter in the reference implementation follows the same skeleton:

1. Optionally filter to a subset (one location, one horizon, one date
   range).
2. Collapse the sample dimension via `mean`, `median`, or `quantile`.
3. Merge forecasts with observations on `(location, time_period)`.
4. Compute the metric or transformation that the plot will display.
5. Build the Altair chart and save the PNG.

Once this skeleton is internalized, new plotters are mostly a matter of
choosing what to compute in step 4 and how to encode it in step 5.


## A catalogue of useful plot types

The four reference plotters cover the three categories established in the
climate-disease forecasting literature: data, evaluation, and prediction.
The plot types below give a broader sense of what fits naturally into the
same pattern.

### 1. Predicted vs observed (scatter)

**Purpose:** Detect bias and outliers at a glance.

- Reduce samples to a point forecast per (location, time_period,
  horizon_distance)
- Merge with observations
- Plot `disease_cases` on x, prediction on y
- Add a 45° reference line for "perfect prediction"

Implemented in the reference project as `ScatterPlotter`.

### 2. Error by horizon (line)

**Purpose:** Show how performance degrades with forecast distance.

- Reduce to point forecast
- Merge with observations
- Compute per-row absolute or squared error
- Aggregate by `horizon_distance`
- Line plot of error vs horizon

This is the example built above.

### 3. Error by location (bar or box)

**Purpose:** Identify which regions are hard to predict.

- Compute per-row error
- Aggregate by `location`
- Bar chart (mean error) or box plot (full distribution of errors)

### 4. Time series with prediction interval (faceted)

**Purpose:** Inspect dynamics and uncertainty side by side.

- Compute median plus lower/upper quantiles per (location, time_period)
- Layer `mark_area` (interval), `mark_line` (median), `mark_line`
  (observation)
- Facet by location with `resolve_scale(y="independent")`

Implemented in the reference project as `EvaluationBackTestPlot`.

### 5. Coverage plot

**Purpose:** Check whether prediction intervals are well calibrated.

- Per row, check if `disease_cases` lies inside the
  [lower quantile, upper quantile] interval
- Compute coverage rate per horizon
- Bar or line plot vs the nominal level (e.g. 0.95)

### 6. Bias plot

**Purpose:** Detect systematic over- or under-prediction.

- Per-row signed error = prediction − truth
- Mean error per horizon or location
- Line or bar plot

### 7. Heatmaps

**Purpose:** Reveal structured patterns across two dimensions.

Examples: horizon × time, location × horizon, calendar week × year. The
reference project's `HeatmapPlotter` is the third variant.

- Aggregate the metric per grid cell
- `mark_rect` with a sequential color scale

### 8. Alert / threshold plots

**Purpose:** Convert a probabilistic forecast into an actionable signal.

- Compute a rolling endemic threshold from observations
- Compute `P(forecast > threshold)` per period from the sample dimension
- Two-panel layout: forecast and threshold on top, exceedance probability
  below

Implemented in the reference project as `OutbreakProbPlotter`.

### 9. Dashboards

**Purpose:** Provide a compact summary across multiple panels.

`chart.save(out_path, scale_factor=2)` accepts anything Altair produces,
including:

- `alt.Chart`
- `LayerChart`
- `FacetChart`
- `VConcatChart`
- `HConcatChart`

Combine 2–4 individual charts with `alt.vconcat`, `alt.hconcat`, or the
`(a + b)` overlay operator before saving.


## Reference implementation

For working examples of every pattern above, see
[Isolated_visualisation](https://github.com/norajeanett/Isolated_visualisation),
which contains four plotters that follow this exact contract:

- `HeatmapPlotter` — small-multiples heatmap (data category)
- `ScatterPlotter` — predicted vs observed (evaluation category)
- `EvaluationBackTestPlot` — faceted backtest with quantile bands
  (evaluation category)
- `OutbreakProbPlotter` — threshold-and-exceedance plot (prediction
  category)

The same project also contains a synthetic data generator
(`generate_realistic_flat_data.py`) that produces realistic flat
forecasts and observations you can use to test new plotters before
plugging them into a real CHAP installation.
