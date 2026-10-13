from flask import Flask, render_template, request
import os, sqlite3
from datetime import datetime
import pandas as pd
import joblib
import requests

app = Flask(__name__)
DATA_FILE, MODEL_FILE, DB_FILE = "Crop_recommendation.csv", "crop_model.joblib", "predictions.db"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def ensure_model():
    if os.path.exists(MODEL_FILE): return
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("Crop_recommendation.csv must be in the same folder as app.py")
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing: raise ValueError(f"Dataset is missing columns: {missing}")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[FEATURES], df["label"])
    joblib.dump(model, MODEL_FILE)

def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS history (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, N REAL, P REAL, K REAL,
          temperature REAL, humidity REAL, ph REAL, rainfall REAL, crop TEXT)''')

def dashboard_data():
    if not os.path.exists(DATA_FILE):
        return dict(crop_labels=[], crop_counts=[], avg_temp=0, avg_humidity=0, avg_rainfall=0, avg_ph=0)
    df = pd.read_csv(DATA_FILE)
    counts = df.label.value_counts().head(12)
    return dict(crop_labels=counts.index.tolist(), crop_counts=counts.values.tolist(),
                avg_temp=round(float(df.temperature.mean()), 1),
                avg_humidity=round(float(df.humidity.mean()), 1),
                avg_rainfall=round(float(df.rainfall.mean()), 1), avg_ph=round(float(df.ph.mean()), 2))



def get_weather(city):
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=20
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            return {"error": f"City '{city}' not found. Try another city."}

        place = results[0]

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
                "timezone": "auto"
            },
            timeout=20
        )
        response.raise_for_status()

        current = response.json().get("current", {})

        return {
            "city": f'{place["name"]}, {place.get("country", "")}',
            "temperature": current.get("temperature_2m", "—"),
            "humidity": current.get("relative_humidity_2m", "—"),
            "rain": current.get("rain", current.get("precipitation", "—")),
            "wind": current.get("wind_speed_10m", "—"),
            "time": current.get("time", "")
        }

    except requests.RequestException as e:
        print(f"Weather API error: {e}")
        return {
            "error": "Weather data could not be loaded. Please try again later."
        }
    except (ValueError, KeyError, TypeError) as e:
        print(f"Weather data error: {e}")
        return {
            "error": "Invalid weather data received. Please try again."
        }
 @app.route("/", methods=["GET", "POST"])
def home():
    prediction = error = None; submitted = {}
    if request.method == "POST":
        try:
            submitted = {k: request.form.get(k, "").strip() for k in FEATURES}
            vals = {k: float(submitted[k]) for k in FEATURES}
            data = pd.DataFrame([[vals[k] for k in FEATURES]], columns=FEATURES)
            prediction = str(model.predict(data)[0])
            with db() as c:
                c.execute("INSERT INTO history (created_at,N,P,K,temperature,humidity,ph,rainfall,crop) VALUES (?,?,?,?,?,?,?,?,?)",
                          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), vals["N"], vals["P"], vals["K"], vals["temperature"], vals["humidity"], vals["ph"], vals["rainfall"], prediction))
        except (ValueError, TypeError, KeyError): error = "Please enter a valid number in every field."
    with db() as c:
        history = c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 8").fetchall()
        total = c.execute("SELECT COUNT(*) AS n FROM history").fetchone()["n"]
    return render_template("index.html", prediction=prediction, error=error, submitted=submitted,
                           history=history, total_predictions=total, dashboard=dashboard_data())

@app.route("/weather", methods=["GET", "POST"])
def weather():
    city = request.values.get("city", "").strip()
    return render_template("weather.html", city=city, weather=get_weather(city) if city else None)

@app.route("/history")
def history_page():
    with db() as c: rows = c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 100").fetchall()
    return render_template("history.html", history=rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
