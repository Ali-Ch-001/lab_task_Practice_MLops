import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_out", type=str, default="model.pkl")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, args.model_out)
    print("Model saved to", args.model_out)
    
    # Create models directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save model to models directory as well (for Flask app)
    joblib.dump(clf, os.path.join(args.model_dir, "house_price_model.pkl"))
    
    # Load original data to get feature names
    iris_df = pd.read_csv(os.path.join(args.data_dir, "iris.csv"))
    feature_columns = [col for col in iris_df.columns if col != 'target']
    
    # Save feature list (ordered)
    joblib.dump(feature_columns, os.path.join(args.model_dir, "model_features.pkl"))
    
    # Create label encoders dictionary (empty for iris since all features are numeric)
    label_encoders = {}
    joblib.dump(label_encoders, os.path.join(args.model_dir, "label_encoders.pkl"))
    
    # Create feature field map (maps feature name to form field name)
    feature_field_map = {feat: feat.replace(" ", "_").replace("(", "").replace(")", "") 
                         for feat in feature_columns}
    joblib.dump(feature_field_map, os.path.join(args.model_dir, "feature_field_map.pkl"))
    
    print(f"All model artifacts saved to {args.model_dir}/")
    print(f"  - house_price_model.pkl")
    print(f"  - model_features.pkl")
    print(f"  - label_encoders.pkl")
    print(f"  - feature_field_map.pkl")
