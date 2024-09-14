import pandas as pd
import joblib


data = pd.read_csv('../dataset/SeoulBikeData.csv', encoding='Windows-1252')
sample = data.sample(20)

model, ref_cols, target = joblib.load("../dataset/ml_model.pkl")

X_new = sample[ref_cols]

X_new = pd.DataFrame(columns=ref_cols)
