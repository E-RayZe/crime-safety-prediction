import os
import sys
import math
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

from custom_linear import CustomLinearRegression
from custom_scaler import CustomStandardScaler
from kmeans_custom import CustomKMeans

DATA_PATH = "dataset/crime_data.csv"
TRAIN_COLUMNS = [
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

def zone_from_cluster_and_total(cluster, total):
    if cluster == 0:
        if total <= 250: return "Safe"
        elif total <= 1500: return "Moderate"
        else: return "Unsafe"
    else:
        if total <= 1650: return "Safe"
        elif total <= 4800: return "Moderate"
        else: return "Unsafe"

MIN_TRAIN_YEARS = 2   
USE_SAVED_MODELS = True   
SAVED_SCALER_PATH = "model/custom_scaler.joblib"
SAVED_KMEANS_PATH = "model/custom_kmeans.joblib"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'YEAR' not in df.columns:
        raise ValueError("YEAR column not found in dataset")
    df['YEAR'] = df['YEAR'].astype(int)
    return df.sort_values(['STATE/UT','DISTRICT','YEAR']).reset_index(drop=True)

def build_time_splits_fixed_year(df, test_year, min_train_years=MIN_TRAIN_YEARS, drop_insufficient=True):
    """
    For each (state,district): train = rows with YEAR < test_year, test = rows with YEAR == test_year
    returns list of dicts with train/test dataframes
    """
    groups = df.groupby(['STATE/UT','DISTRICT'])
    splits = []
    for (s,d), g in groups:
        g = g.sort_values('YEAR')
        train = g[g['YEAR'] < test_year]
        test = g[g['YEAR'] == test_year]
        if test.empty:
            continue
        if len(train) < min_train_years:
            if drop_insufficient:
                continue
        splits.append({
            'state': s, 'district': d, 'train': train.reset_index(drop=True), 'test': test.reset_index(drop=True), 'test_year': test_year
        })
    return splits

def evaluate(test_year=2012):
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Total rows: {len(df)}")

    print(f"Building splits for test_year={test_year} ...")
    splits = build_time_splits_fixed_year(df, test_year=test_year, min_train_years=MIN_TRAIN_YEARS, drop_insufficient=True)
    print(f"Total districts available for evaluation (test_year={test_year}): {len(splits)}")
    if len(splits) == 0:
        print("No splits available. Make sure dataset has rows with the requested test year and sufficient training years.")
        return

    X_train_rows = []
    for s in splits:
        t = s['train']
        if len(t) > 0:
            X_train_rows.append(t[TRAIN_COLUMNS].values)
    if not X_train_rows:
        print("No training rows found to build global scaler/kmeans.")
        return
    X_train_all = np.vstack(X_train_rows)

    if USE_SAVED_MODELS and os.path.exists(SAVED_SCALER_PATH) and os.path.exists(SAVED_KMEANS_PATH):
        print("Loading saved scaler and kmeans models...")
        scaler = joblib.load(SAVED_SCALER_PATH)
        kmeans = joblib.load(SAVED_KMEANS_PATH)
    else:
        print("Training custom scaler and KMeans on training pool...")
        scaler = CustomStandardScaler()
        scaler.fit(X_train_all)
        X_train_scaled = scaler.transform(X_train_all)

        kmeans = CustomKMeans(n_clusters=3, max_iters=200, random_state=42)
        kmeans.fit(X_train_scaled)

        os.makedirs("model", exist_ok=True)
        joblib.dump(scaler, SAVED_SCALER_PATH)
        joblib.dump(kmeans, SAVED_KMEANS_PATH)
        print(f"Saved scaler -> {SAVED_SCALER_PATH}, kmeans -> {SAVED_KMEANS_PATH}")

    per_col_true_pred = defaultdict(list)
    actual_zone_list = []
    predicted_zone_list = []
    per_district_rows = []

    print("Starting per-district evaluation...")

    for entry in splits:
        s_name = entry['state']
        d_name = entry['district']
        train_df = entry['train']
        test_df = entry['test']   
        test_year = entry['test_year']

        predicted = {}
        for col in TRAIN_COLUMNS:
            tt = train_df[['YEAR', col]].dropna()
            if len(tt) < 2:
                val = float(tt[col].iloc[-1]) if len(tt) else 0.0
                pred_val = val
            else:
                model = CustomLinearRegression()
                try:
                    model.fit(tt['YEAR'].values.reshape(-1,1), tt[col].values)
                    pred_val = float(model.predict([test_year])[0])
                except Exception as e:
                    pred_val = float(tt[col].iloc[-1])
            if math.isnan(pred_val) or pred_val is None:
                pred_val = 0.0
            pred_val = max(0.0, pred_val)
            predicted[col] = pred_val

            actual_val = float(test_df[col].iloc[0]) if (col in test_df.columns and not pd.isna(test_df[col].iloc[0])) else 0.0
            per_col_true_pred[col].append((actual_val, pred_val))

        actual_vec = test_df[TRAIN_COLUMNS].iloc[0].astype(float).values
        pred_vec = np.array([predicted[c] for c in TRAIN_COLUMNS], dtype=float)

        actual_scaled = scaler.transform(actual_vec.reshape(1, -1))
        pred_scaled = scaler.transform(pred_vec.reshape(1, -1))

        try:
            pred_cluster = int(kmeans.predict(pred_scaled)[0])
            actual_cluster = int(kmeans.predict(actual_scaled)[0])
        except Exception as e:
            pred_cluster = 0
            actual_cluster = 0

        total_actual = float(actual_vec.sum())
        total_predicted = float(pred_vec.sum())

        zone_actual = zone_from_cluster_and_total(actual_cluster, total_actual)
        zone_predicted = zone_from_cluster_and_total(pred_cluster, total_predicted)

        actual_zone_list.append(zone_actual)
        predicted_zone_list.append(zone_predicted)

        per_district_rows.append({
            'STATE': s_name, 'DISTRICT': d_name, 'TEST_YEAR': test_year,
            'ACTUAL_ZONE': zone_actual, 'PREDICTED_ZONE': zone_predicted,
            'TOTAL_ACTUAL': total_actual, 'TOTAL_PREDICTED': total_predicted
        })

    print("\nComputing regression metrics per crime category...")
    reg_rows = []
    for col, pairs in per_col_true_pred.items():
        y_true = np.array([p[0] for p in pairs])
        y_pred = np.array([p[1] for p in pairs])
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
        reg_rows.append({'feature': col, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    reg_df = pd.DataFrame(reg_rows).sort_values('MAE')
    reg_df.to_csv("evaluation_regression_metrics.csv", index=False)
    print("Saved regression metrics -> evaluation_regression_metrics.csv")

    print("\nComputing classification metrics for zones...")
    zone_acc = accuracy_score(actual_zone_list, predicted_zone_list)
    class_report = classification_report(actual_zone_list, predicted_zone_list, zero_division=0)
    conf_mat = confusion_matrix(actual_zone_list, predicted_zone_list, labels=["Safe","Moderate","Unsafe"])
    print(f"Zone Accuracy: {zone_acc:.4f}\n")
    print("Classification Report:\n", class_report)
    print("Confusion Matrix (rows=actual, cols=predicted) [Safe, Moderate, Unsafe]:\n", conf_mat)

    pd.DataFrame(per_district_rows).to_csv("evaluation_zone_classification.csv", index=False)
    print("Saved zone classification results -> evaluation_zone_classification.csv")

    print("\nEvaluation completed.")

if __name__ == "__main__":
    TEST_YEAR = 2012
    evaluate(test_year=TEST_YEAR)
