import os
import mlflow
import argparse
import time
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error



def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
       df =  pd.read_csv(url, sep=";")
       return df
    except Exception as e:
       raise e
  
def eval(actual, pred) :
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 =  r2_score(actual, pred)
      
    return rmse, mae, r2
   


def main(alpha, l1_ratio):
    df = load_data()
    TARGET = "quality"
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ML-Model-WineQuality-1")
    with mlflow.start_run():
        mlflow.log_param("Alpha", alpha)
        mlflow.log_param("l1-ratio", l1_ratio)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse, mae, r2= eval(y_test, y_pred)
        
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "trained_model") # model, foldername
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(f"Artifact created at {time.asctime()}")
        mlflow.log_artifacts("dummy")
        
        
        
        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=int, default=0.2)
    args.add_argument("--l1_ratio", "-l1", type=int, default=0.3)
    parser_args = args.parse_args()
    
    main(parser_args.alpha, parser_args.l1_ratio)