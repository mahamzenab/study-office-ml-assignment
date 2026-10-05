import json
import numpy as np
import pandas as pd
import xgboost as xgb

class PortableRiskModel:
    def __init__(self, model_dir="model"):
        with open(f"{model_dir}/preprocess.json", encoding="utf-8") as f:
            self.info=json.load(f)
        self.booster=xgb.Booster()
        self.booster.load_model(f"{model_dir}/booster.json")
        self.features=self.info["features"]
        self.num_cols=self.info["numeric_columns"]
        self.cat_cols=self.info["categorical_columns"]

    def transform(self, df):
        parts=[]
        for c in self.num_cols:
            s=pd.to_numeric(df[c], errors="coerce").fillna(self.info["numeric_medians"][c]).to_numpy(dtype=float).reshape(-1,1)
            parts.append(s)
        for c in self.cat_cols:
            cats=self.info["categories"][c]
            vals=df[c].astype(str).to_numpy()
            mat=np.zeros((len(df),len(cats)),dtype=float)
            index={v:i for i,v in enumerate(cats)}
            for r,v in enumerate(vals):
                if v in index: mat[r,index[v]]=1.0
            parts.append(mat)
        return np.hstack(parts)

    def predict_risk(self, df):
        X=self.transform(df[self.features])
        return self.booster.predict(xgb.DMatrix(X))
