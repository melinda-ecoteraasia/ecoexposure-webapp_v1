from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES=[
    "Clarity_Raw","ROI_Mean_Raw","ROI_SD_Raw","Center_Clarity_Raw",
    "Mid_Clarity_Raw","Outer_Clarity_Raw","Center_Mid_Contrast_Raw",
    "Center_Outer_Contrast_Raw","RCDI_Raw","COR_Raw","Center_Clarity_Z",
    "Center_Mid_Contrast_Z","Center_Outer_Contrast_Z","Center_GEC","LuED",
    "Radial_Gradient","RSI_Correlation","RSI_RadialCV",
]
VALID=[0,10,25,50,75,100]

BASE=Path(r"C:\Users\micro\OneDrive\Documents\MICROPLASTICS DATA")
FILTERED_CSV=BASE/"_EcoExposure_BeakerRelative_ROI_V2Detection"/"Historical_30min_MP_ROI60.csv"
SALT_CSV=BASE/"_EcoExposure_30min_Analysis"/"EcoExposure_30min_MP_ONLY_Salt.csv"
OUT_DIR=Path(__file__).resolve().parent/"models"
OUT_DIR.mkdir(parents=True,exist_ok=True)

def make_model():
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("rf",RandomForestClassifier(
            n_estimators=1000,class_weight="balanced",random_state=42,
            min_samples_leaf=3,max_features="sqrt"
        ))
    ])

def train_one(csv_path,model_name,output_name):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing training CSV: {csv_path}")
    df=pd.read_csv(csv_path)
    df["Concentration_MP"]=pd.to_numeric(df["Concentration_MP"],errors="coerce")
    df=df[df["Concentration_MP"].isin(VALID)].copy()
    missing=[f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing features: {missing}")
    X=df[FEATURES]; y=df["Concentration_MP"].astype(int)
    model=make_model(); model.fit(X,y)
    bundle={
        "model":model,"model_name":model_name,"model_version":model_name,
        "feature_names":FEATURES,"class_labels":sorted(y.unique().tolist()),
        "training_rows":len(df),"training_csv":csv_path.name,
    }
    out=OUT_DIR/output_name
    joblib.dump(bundle,out)
    print(f"Saved: {out}")
    print(f"  rows: {len(df)}")
    print(f"  classes: {sorted(y.unique().tolist())}")

if __name__=="__main__":
    train_one(FILTERED_CSV,"filtered_mp_dev_v1","filtered_rf_v1.joblib")
    train_one(SALT_CSV,"salt_mp_v1","salt_rf_v1.joblib")
    print("\nUpload these to GitHub under inference/models/:")
    print("  filtered_rf_v1.joblib")
    print("  salt_rf_v1.joblib")
