# -*- coding: utf-8 -*-
# Library import
import pandas as pd
import numpy as np
import sys
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb

# load data from pkl file
data = pd.read_pickle("../dataset/processed_data.pkl")
target = "Rented Bike Count"

# preparation of train and test set
X = data.drop('Rented Bike Count', axis=1)
y = data['Rented Bike Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Feature transformation (pipeline definition)
numeric_features = X.select_dtypes(include=["float", "int64", "UInt32"]).columns
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_features = X.select_dtypes(include=["category"]).columns
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Preprocesor definition
preprocesor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# setup pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocesor), ("regressor", lgb.LGBMRegressor(random_state=42, verbose=0))]
)

# fit the pipeline to train the model on the training set
model = pipeline.fit(X_train, y_train)


#Model evaluation
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R2:", r2)

#Model export
ref_cols = list(X.columns)
joblib.dump(value=[model, ref_cols, target], filename="../dataset/ml_model.pkl")
