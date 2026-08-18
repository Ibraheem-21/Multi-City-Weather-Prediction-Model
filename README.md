# Multi-City Weather Prediction Model

Predict tomorrow’s maximum temperature (°F) with Ridge regression across multiple cities.

- **Oakland** uses the bundled NOAA airport CSV in this repo
- **Other cities** download [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) station history (Open-Meteo archive is the fallback)

Included cities: Oakland, Toronto, San Francisco, New York, Chicago, Vancouver, Seattle.

**Live demo:** [https://multi-city-weather-prediction-model.streamlit.app/](https://multi-city-weather-prediction-model.streamlit.app/)

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

### Optional: AI insights (Gemini)

Enhanced forecast narratives and the **Ask** chat tab use Google Gemini when a key is configured. Without a key, the app falls back to built-in Ridge-based explanations.

**Local:** copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and set:

```toml
GEMINI_API_KEY = "your-key-from-ai-studio"
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey). New keys use the `AQ.*` format — that is expected.

**Test connectivity:**

```bash
python scripts/test_gemini.py
```

## Deploy demo (Streamlit Community Cloud)

**Live app:** [https://multi-city-weather-prediction-model.streamlit.app/](https://multi-city-weather-prediction-model.streamlit.app/)

1. Push this repo to GitHub (models/metrics/predictions under `artifacts/` are included for fast cold starts).
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → select this repo → branch `main` → main file `app.py`.
4. Click **Deploy**. You’ll get a public URL like `https://….streamlit.app`.
5. **Optional — AI on the live demo:** App settings → **Secrets** → add `GEMINI_API_KEY`, then reboot the app.

Notes:

- Oakland works immediately from the bundled CSV.
- Other cities may download NOAA history on first visit (can take ~30–60s once, then cached on that instance).
- Prefers committed `artifacts/*_ridge.joblib` so the app does not retrain on every boot.
- Optional: set a custom subdomain in the Streamlit Cloud app settings.
- Never commit `.streamlit/secrets.toml` — it is gitignored.

## What you get in the UI

- **Next-day forecast** for the selected city
- **Forecast insight** — Ridge feature contribution chart plus narrative (Gemini when configured)
- **°F / °C toggle**
- **Forecast what-if** — simulate different precip / TMAX / TMIN and see the model’s next-day prediction
- **Predict by date** — pick a month and day (e.g. June 25); see typical prediction and year-by-year actuals
- **Performance** — actual vs predicted, RMSE / MAE, worst misses
- **Insights** — coefficients, correlations
- **History** — temperature and precipitation charts
- **About the Model** — model card, feature dictionary, limitations, cross-city RMSE comparison
- **Ask** — chat about forecasts, drivers, and accuracy (Gemini when configured; rule-based fallback otherwise)

## Project layout

| Path | Role |
|------|------|
| `Oakland_Weather.csv` | Bundled Oakland NOAA daily observations |
| `src/` | Data loading, features, model, insights, AI helpers |
| `src/llm_providers.py` | Gemini / Groq / OpenRouter / OpenAI clients (auto-detect keys) |
| `src/prediction_explainer.py` | Ridge decomposition + optional LLM narrative |
| `src/model_card.py` | Model documentation and cross-city metrics |
| `src/ai_assistant.py` | Rule-based Q&A fallback |
| `train.py` | Train and save models under `artifacts/` |
| `app.py` | Streamlit UI |
| `scripts/test_gemini.py` | Quick Gemini API connectivity check |
| `data/` | Local cache for remote downloads (gitignored) |
| `artifacts/` | Saved models, metrics, test predictions |
| `.streamlit/config.toml` | Streamlit Cloud / local UI defaults |
| `.streamlit/secrets.toml.example` | Template for optional API keys |
| `Oakland_CA_Weather_Model.ipynb` | Original exploration notebook (runs locally) |

## Model notes

- Target: next day’s `temp_max`
- Features: `precip`, `temp_max`, `temp_min`, 30-day `month_day_max`, `max_min`
- Train / test: through 2020 vs from 2021 onward
- Units: °F and inches (UI can display Celsius)
- Explanations: linear Ridge contributions (coefficient × feature value) plus optional LLM summary

## Notebook

Open `Oakland_CA_Weather_Model.ipynb` and run all cells. It reads `Oakland_Weather.csv` directly (no Colab upload).
