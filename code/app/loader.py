import os
import joblib

def load_models(cls_path,reg_path):
    if not os.path.exists(cls_path):
        raise FileNotFoundError(f"Classifier not Found {cls_path}")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regressor not Found {reg_path}")
    cls=joblib.load(cls_path)
    reg=joblib.load(reg_path)

    return cls,reg
