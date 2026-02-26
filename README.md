# Creating a New Custom Backtest Visualization in CHAP

This guide walks you through adding a simple custom backtest plot to CHAP, verifying that it works locally, and understanding how to extend the approach to other types of visualizations.

It is written as a practical, step-by-step tutorial. 


## Add Your First Backtest Plot
What You Are Building

A custom backtest visualization in CHAP is:

- A Python class

- Decorated with `@backtest_plot` 

- Subclassing `BacktestPlotBase`

- Returning an Altair chart

- Automatically discoverable by the CLI and Modeling App

Once registered, your plot will be available via:
```python
chap plot-backtest evaluation.nc output.html --plot-type <plot_id>
```
And through the REST API. 

## Understand the data You Recieve 
Your plot receives standardized flat DataFrames.

Observations 

| Column          | Description         |
| --------------- | ------------------- |
| `location`      | Location identifier |
| `time_period`   | Time key            |
| `disease_cases` | Observed value      |


Forcasts 
| Column             | Description                            |
| ------------------ | -------------------------------------- |
| `location`         | Location identifier                    |
| `time_period`      | Forecasted period                      |
| `horizon_distance` | Steps ahead                            |
| `sample`           | Sample index (probabilistic forecasts) |
| `forecast`         | Predicted value                        |

Your job is to transform these into a meaningful visualization.


## Create a new plot file 
Create a new file inside: 
```python
chap_core/assessment/backtest_plots/
```

Example: 
```python
mae_by_horizon_plot.py
```

### Implement a plot 
Below is a simple example: Mean Absolute Error (MAE) by horizon distance.

This is a good first plot because it:

- Uses standard columns

- Requires minimal transformation

- Demonstrates the full plugin system

```python
from typing import Optional
import pandas as pd
import altair as alt

from chap_core.assessment.backtest_plots import (
    backtest_plot,
    BacktestPlotBase,
    ChartType,
)

@backtest_plot(
    plot_id="mae_by_horizon",
    name="MAE by horizon",
    description="Mean absolute error as a function of horizon_distance.",
)
class MaeByHorizonPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: Optional[pd.DataFrame] = None,
    ) -> ChartType:

        # 1. Collapse probabilistic samples to a point forecast (median)
        point_forecasts = (
            forecasts
            .groupby(["location", "time_period", "horizon_distance"])["forecast"]
            .median()
            .reset_index()
            .rename(columns={"forecast": "prediction"})
        )

        # 2. Join predictions with observations
        merged = point_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]],
            on=["location", "time_period"],
            how="left",
        )

        # 3. Compute absolute error
        merged["abs_error"] = (
            merged["prediction"] - merged["disease_cases"]
        ).abs()

        # 4. Aggregate error by horizon
        mae = (
            merged.groupby("horizon_distance")["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"abs_error": "mae"})
        )

        # 5. Create Altair chart
        return (
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
                title="Mean Absolute Error by Horizon",
            )
        )
```


## Registrer the plot 
CHAP discovers plots when their modules are imported.

Add your module to ```_discover_plots()``` inside:
```python
chap_core/assessment/backtest_plots/__init__.py
```

Example: 
```python
def _discover_plots():
    from chap_core.assessment.backtest_plots import (
        metrics_dashboard,
        sample_bias_plot,
        evaluation_plot,
        mae_by_horizon_plot,  # <-- add this line
    )
```

If you forget this step, your plot will not appear.


## Test it

**Option A**
_Unit Test with Flat Fixtures_

Create a test similar to existing backtest plot tests:

```python
def test_mae_by_horizon_plot(flat_observations, flat_forecasts):
    import pandas as pd
    from chap_core.assessment.backtest_plots.mae_by_horizon_plot import MaeByHorizonPlot

    plot = MaeByHorizonPlot()
    chart = plot.plot(
        pd.DataFrame(flat_observations),
        pd.DataFrame(flat_forecasts),
    )

    assert chart is not None
```

This verifies:

- The transformation runs

- An Altair chart is returned

**Option B**
_Run Through CLI_

Generate a evaluation:
```python
chap eval my_model data.csv evaluation.nc
```
Generate your plot: 
```python
chap plot-backtest evaluation.nc mae.html --plot-type mae_by_horizon
```

If successful, you now have a working custom visualization.

## Verify in the Modeling App
Once registered, your plot becomes available automatically through:
```python
GET /visualization/backtest-plots/
GET /visualization/backtest-plots/{plot_id}/{backtest_id}
```

No frontend changes are required.

## How to Think About Backtest Visualizations in CHAP

Most backtest plots follow the same technical structure:

1. Optionally filter data (location, horizon, time range)
2. Collapse sample dimension (median, mean, quantiles)
3. Merge forecasts with observations
4. Compute metrics or transformations
5. Build Altair chart (single, layered, faceted, or concatenated)

Once you understand this pattern, implementing new visualizations becomes straightforward





