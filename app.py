from fastapi import FastAPI, UploadFile, File
import pandas as pd
from sklearn.ensemble import IsolationForest
import tempfile
import os

app = FastAPI(title="Aadhaar Fraud Sentinel ML Service")

@app.get("/")
def health():
    return {"status": "ML service running"}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    # 1️⃣ Save uploaded CSV to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 2️⃣ Load CSV
        df = pd.read_csv(tmp_path)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # 3️⃣ Basic validation
        required_cols = {"date", "district"}
        if not required_cols.issubset(df.columns):
            return {
                "error": "CSV must contain 'date' and 'district' columns"
            }

        # 4️⃣ Aggregate daily counts
        df_daily = (
            df.groupby(["date", "district"])
              .size()
              .reset_index(name="daily_count")
        )

        # 5️⃣ Train Isolation Forest on THIS CSV
        model = IsolationForest(contamination=0.02, random_state=42)
        df_daily["anomaly"] = model.fit_predict(df_daily[["daily_count"]])

        df_daily["status"] = df_daily["anomaly"].map({
            1: "Normal",
            -1: "Suspicious"
        })

        # 6️⃣ Summary
        suspicious = df_daily[df_daily["status"] == "Suspicious"]

        response = {
            "total_rows": int(len(df)),
            "total_days_districts": int(len(df_daily)),
            "suspicious_count": int(len(suspicious)),
            "top_suspicious": suspicious
                .sort_values("daily_count", ascending=False)
                .head(10)
                .to_dict(orient="records")
        }

        return response

    finally:
        # 7️⃣ Cleanup temp file
        os.remove(tmp_path)
