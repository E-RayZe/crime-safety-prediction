import pandas as pd
import numpy as np
import joblib
from custom_scaler import CustomStandardScaler     
from kmeans_custom import CustomKMeans            

df = pd.read_csv("dataset/crime_data.csv")

train_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER',
    'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE', 'KIDNAPPING & ABDUCTION',
    'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS', 'KIDNAPPING AND ABDUCTION OF OTHERS',
    'DACOITY', 'PREPARATION AND ASSEMBLY FOR DACOITY', 'ROBBERY', 'BURGLARY',
    'THEFT', 'AUTO THEFT', 'OTHER THEFT', 'RIOTS', 'CRIMINAL BREACH OF TRUST',
    'CHEATING', 'COUNTERFIETING', 'ARSON', 'HURT/GREVIOUS HURT', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY', 'INSULT TO MODESTY OF WOMEN',
    'CRUELTY BY HUSBAND OR HIS RELATIVES', 'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
    'CAUSING DEATH BY NEGLIGENCE', 'OTHER IPC CRIMES'
]

X = df[train_columns].values

scaler = CustomStandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = CustomKMeans(n_clusters=3, max_iters=200, random_state=42)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.labels_

joblib.dump(kmeans, "model/custom_kmeans.joblib")
joblib.dump(scaler, "model/custom_scaler.joblib")

df.to_csv("dataset/crime_data_clustered.csv", index=False)

print("✅ Custom KMeans model trained successfully using Custom Scaler!")
