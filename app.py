from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

from custom_linear import CustomLinearRegression
from kmeans_custom import CustomKMeans

app = Flask(__name__)

df = pd.read_csv("dataset/crime_data.csv")
train_columns = [
    'MURDER','ATTEMPT TO MURDER','CULPABLE HOMICIDE NOT AMOUNTING TO MURDER',
    'RAPE','CUSTODIAL RAPE','OTHER RAPE','KIDNAPPING & ABDUCTION',
    'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS','KIDNAPPING AND ABDUCTION OF OTHERS',
    'DACOITY','PREPARATION AND ASSEMBLY FOR DACOITY','ROBBERY','BURGLARY',
    'THEFT','AUTO THEFT','OTHER THEFT','RIOTS','CRIMINAL BREACH OF TRUST',
    'CHEATING','COUNTERFIETING','ARSON','HURT/GREVIOUS HURT','DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY','INSULT TO MODESTY OF WOMEN',
    'CRUELTY BY HUSBAND OR HIS RELATIVES','IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
    'CAUSING DEATH BY NEGLIGENCE','OTHER IPC CRIMES'
]

scaler = joblib.load("model/custom_scaler.joblib")
kmeans = joblib.load("model/custom_kmeans.joblib")

@app.route('/')
def home():
    return render_template(
        "index.html",
        states=sorted(df['STATE/UT'].unique()),
        districts=[],
        crimes=train_columns,
        table_data=None,
        bar_chart_data=None,
        crime_trend_data=None,
        result=None,
        color=None
    )

@app.route('/predict', methods=['POST'])
def predict():

    state = request.form['state']
    district = request.form['district']
    year = int(request.form['year'])

    dsub = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]
    if dsub.empty:
        return render_template("index.html", result="No Data Found", color="#9e9e9e")

    predicted = {}

    for col in train_columns:
        hist = dsub[['YEAR', col]].dropna()

        if len(hist) < 2:
            predicted[col] = int(hist[col].iloc[-1]) if len(hist) else 0
        else:
            lr = CustomLinearRegression()
            lr.fit(hist['YEAR'].values, hist[col].values)
            val = lr.predict([year])[0]
            predicted[col] = max(0, int(val))

    record = pd.DataFrame([predicted])
    pred_vec = record.iloc[0].values.astype(float)

    pred_scaled = scaler.transform(pred_vec.reshape(1, -1))
    cluster = int(kmeans.predict(pred_scaled)[0])

    total_crime = pred_vec.sum()
    cluster_intensity = kmeans.centroids[cluster].mean()

    if cluster_intensity < -0.6 and total_crime < 2000:
        zone = "Safe Zone"
        color = "#4CAF50"

    elif cluster_intensity < 0.4 and total_crime < 6000:
        zone = "Moderate Zone"
        color = "#FF9800"

    else:
        zone = "Unsafe Zone"
        color = "#F44336"

    result_text = f"{zone} (Predicted for {year})"

    crime_trend_data = {}

    for col in train_columns:
        years = list(range(2001, 2027))
        values = []

        hist = dsub[['YEAR', col]].dropna()
        lr = CustomLinearRegression()

        if len(hist) >= 2:
            lr.fit(hist['YEAR'].values, hist[col].values)

        for y in years:
            if y in hist['YEAR'].values:
                values.append(int(hist[hist['YEAR'] == y][col].iloc[0]))
            else:
                if len(hist) >= 2:
                    values.append(max(0, int(lr.predict([y])[0])))
                else:
                    values.append(0)

        crime_trend_data[col] = {"years": years, "values": values}

    return render_template(
        "index.html",
        states=sorted(df['STATE/UT'].unique()),
        districts=sorted(df[df['STATE/UT'] == state]['DISTRICT'].unique()),
        crimes=train_columns,
        table_data=record.iloc[0].to_dict(),
        bar_chart_data=record.iloc[0].to_dict(),
        crime_trend_data=crime_trend_data,
        result=result_text,
        color=color
    )

@app.route('/get_districts', methods=['POST'])
def get_districts():
    state = request.form['state']
    return jsonify({
        "districts": sorted(df[df['STATE/UT'] == state]['DISTRICT'].unique())
    })

if __name__ == "__main__":
    app.run(debug=True)
