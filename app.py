from fastapi import FastAPI, UploadFile, File
import pandas as pd
from sklearn.ensemble import IsolationForest
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Aadhaar Fraud Sentinel ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ML service running"}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)
        df.columns = df.columns.str.strip().str.lower()

        if not {"date", "district"}.issubset(df.columns):
            return {"error": "CSV must contain 'date' and 'district'"}

        # Notebook-style aggregation
        df_daily = (
            df.groupby(["date", "district"])
              .size()
              .reset_index(name="Daily_Count")
        )

        # NOTEBOOK MODEL (no random_state)
        model = IsolationForest(contamination=0.02)
        df_daily["Anomaly_Score"] = model.fit_predict(df_daily[["Daily_Count"]])
        df_daily["Status"] = df_daily["Anomaly_Score"].map({
            1: "Normal",
            -1: "Suspicious"
        })

        # Notebook growth logic (no cleanup)
        df_daily["Growth_Rate"] = (
            df_daily
            .groupby("district")["Daily_Count"]
            .pct_change()
        )

        # Notebook future alerts logic
        future_alerts = (
            df_daily[df_daily["Growth_Rate"] > 0.5]
            .drop_duplicates("district")
        )

        return {
            "total_rows": int(len(df)),
            "total_days_districts": int(len(df_daily)),
            "suspicious_count": int(
                len(df_daily[df_daily["Status"] == "Suspicious"])
            ),

            "top_suspicious": (
                df_daily[df_daily["Status"] == "Suspicious"]
                .sort_values("Daily_Count", ascending=False)
                .head(15)
                .to_dict(orient="records")
            ),

            "future_alerts": future_alerts[[
                "district",
                "date",
                "Daily_Count",
                "Growth_Rate"
            ]].to_dict(orient="records"),

            "scatter": df_daily[[
                "date",
                "Daily_Count",
                "Status"
            ]].to_dict(orient="records")
        }

    finally:
        os.remove(tmp_path)
