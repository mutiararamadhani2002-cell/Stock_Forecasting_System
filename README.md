# 📈 StockSARIMA × Prophet

Aplikasi forecasting harga saham IDX menggunakan model **SARIMA** dan **Prophet** dengan ensemble prediction.

---

## 🗂️ Struktur Proyek

```
stock-forecasting/
├── app.py              # Streamlit dashboard utama
├── data_pipeline.py    # Download & cache data per ticker
├── model_utils.py      # Inference, ensemble, metrics, signals
├── train.py            # Training pipeline SARIMA + Prophet
├── models/             # Model tersimpan per ticker (.pkl)
├── data/               # Data CSV per ticker
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Pakai

### 1. Jalankan aplikasi
```bash
streamlit run app.py
```

### 2. Training model via CLI
```bash
# Train semua ticker
python train.py

# Train ticker tertentu
python train.py "ANTM - PT Aneka Tambang Tbk"
```

---

## 🔑 Fitur Utama

- **Multi-ticker support** — ANTM, BBCA, BBRI, BMRI, TLKM, dll.
- **SARIMA** dilatih pada `log1p(Close)` → tidak ada overflow
- **Prophet** menangani seasonality mingguan & tahunan
- **Ensemble** weighted average (configurable)
- **Sinyal trading** BUY / SELL / HOLD per model
- **Metrics** MAE, RMSE, MAPE yang akurat

---

## 🛡️ Fix Utama

| Masalah | Solusi |
|---|---|
| SARIMA overflow (`e+29`) | Training pada `log1p`, inverse `expm1`, drift clamp |
| Hanya 1 ticker didownload | `data_pipeline.py` dinamis per ticker |
| Model tidak tersimpan per ticker | Disimpan sebagai `models/{ticker}_sarima.pkl` |
| Ensemble rusak | NaN-aware weighted average |
| MAE/RMSE sangat besar | Alignment index yang benar |

---

## ⚠️ Disclaimer

Aplikasi ini hanya untuk tujuan edukasi. Jangan gunakan prediksi ini sebagai satu-satunya dasar keputusan investasi.
