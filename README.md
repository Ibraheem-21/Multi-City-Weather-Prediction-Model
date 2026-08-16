# Multi-City Weather Prediction Model

Predict tomorrow’s maximum temperature (°F) with Ridge regression across multiple cities.

- **Oakland** uses the bundled NOAA airport CSV in this repo
- **Other cities** download [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) station history (Open-Meteo archive is the fallback)

Included cities: Oakland, Toronto, San Francisco, New York, Chicago, Vancouver, Seattle.

## Quick start

```bash
pip install -r requirements.txt

# Train all cities (downloads remote history on first run)
python train.py

# Or train a subset
python train.py --city oakland --city toronto

# Launch the UI
streamlit run app.py
```

List cities:

```bash
python train.py --list
```

## What you get in the UI

- **Next-day forecast** for the selected city
- **What-if controls** for precip / TMAX / TMIN
- **Test performance** (actual vs predicted, RMSE / MAE, worst misses)
- **Insights** (coefficients, correlations)
- **History** charts

## Project layout

| Path | Role |
|------|------|
| `Oakland_Weather.csv` | Bundled Oakland NOAA daily observations |
| `src/` | Data loading, features, model, insights |
| `train.py` | Train and save models under `artifacts/` |
| `app.py` | Streamlit UI |
| `data/` | Cached remote downloads |
| `artifacts/` | Saved models, metrics, test predictions |
| `Oakland_CA_Weather_Model.ipynb` | Original exploration notebook (runs locally) |

## Model notes

- Target: next day’s `temp_max`
- Features: `precip`, `temp_max`, `temp_min`, 30-day `month_day_max`, `max_min`
- Train / test: through 2020 vs from 2021 onward
- Units: °F and inches

## Notebook

Open `Oakland_CA_Weather_Model.ipynb` and run all cells. It reads `Oakland_Weather.csv` directly (no Colab upload).
