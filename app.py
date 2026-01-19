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
    # 1️⃣ Save uploaded CSV temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 2️⃣ Read CSV
        df = pd.read_csv(tmp_path)
        df.columns = df.columns.str.strip().str.lower()

        # 3️⃣ Validate required columns
        if not {"date", "district"}.issubset(df.columns):
            return {"error": "CSV must contain 'date' and 'district'"}

        # 4️⃣ Aggregate daily counts
        df_daily = (
            df.groupby(["date", "district"])
              .size()
              .reset_index(name="daily_count")
        )

        # 🔹 IMPORTANT: sort for growth calculation
        df_daily = df_daily.sort_values(["district", "date"])

        # 5️⃣ Isolation Forest (Anomaly Detection)
        model = IsolationForest(contamination=0.02, random_state=42)
        df_daily["anomaly"] = model.fit_predict(df_daily[["daily_count"]])
        df_daily["status"] = df_daily["anomaly"].map({
            1: "Normal",
            -1: "Suspicious"
        })

        # 6️⃣ Growth Rate Calculation (NEW)
        df_daily["growth_rate"] = (
            df_daily
            .groupby("district")["daily_count"]
            .pct_change()
        )

        df_daily["growth_rate"] = (
        df_daily["growth_rate"]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
        )
        
        # 7️⃣ Future Risk Alerts (NEW)
        future_alerts = (
            df_daily[df_daily["growth_rate"] > 0.5]
            .dropna(subset=["growth_rate"])
            .sort_values("growth_rate", ascending=False)
            .head(15)
        )

        # 8️⃣ Prepare response for frontend
        response = {
            "total_rows": int(len(df)),
            "total_days_districts": int(len(df_daily)),
            "suspicious_count": int(
                len(df_daily[df_daily["status"] == "Suspicious"])
            ),

            "top_suspicious": (
                df_daily[df_daily["status"] == "Suspicious"]
                .sort_values("daily_count", ascending=False)
                .head(10)
                .to_dict(orient="records")
            ),

            "future_alerts": future_alerts[[
                "district",
                "date",
                "daily_count",
                "growth_rate"
            ]].to_dict(orient="records"),

            "scatter": df_daily[[
                "date",
                "daily_count",
                "status"
            ]].to_dict(orient="records")
        }

        return response

    finally:
        os.remove(tmp_path)
