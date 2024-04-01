import os
import mlflow
import argparse
import time
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
import uuid
from datetime import datetime, timedelta


time_day = datetime.utcnow() + timedelta(hours=3)
formatted_time_day  = time_day.strftime('%Y-%m-%d-%H-%M')
unique_id = str(uuid.uuid4().hex)[:6]
unique_format = formatted_time_day + '-' + unique_id
run_name_model_version = "trained-V" + unique_format
print("Run name: ", run_name_model_version)



def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
       df =  pd.read_csv(url, sep=";")
       return df
    except Exception as e:
       raise e
  
def eval(actual, pred) :
    # rmse = mean_squared_error(actual, pred, squared=False)
    rmse = mean_squared_error(actual, pred)
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
    mlflow.set_experiment("Red-Wine-Model")
    with mlflow.start_run(run_name = run_name_model_version):
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
        mlflow.sklearn.save_model(model,
                                  os.path.join("Model" ,
                                               "Artifactcs",
                                               unique_format))# Change my_model to path of your choice to save the model
        
        
        
        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=0.2)
    args.add_argument("--l1-ratio", "-l1", type=float, default=0.3)
    parser_args = args.parse_args()
    
    main(parser_args.alpha, parser_args.l1_ratio)