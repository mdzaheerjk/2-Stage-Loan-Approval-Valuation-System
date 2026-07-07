from app.loader import load_models
from app.predict import two_stage_predict
from app.utils import build_application_from_dict
import yaml

config=yaml.safe_load(open("config.yaml"))

cls,reg=load_models(config)

def run_cli():
    data={
        'no_of_dependents':input("Enter The Number of Dependents"),
        'education':'Graduate',
        'self_employed':'No',
        'income_annum':1200000,
        'loan_ammount':30000,
        'loan_term':12,
        'cibil_score':8,
        'residential_assets_value': 2000000,
        'commercial_assets_value': 2000000,
        'luxury_assets_value':0,
        'bank_assets_value': 55000
    }
    df=build_application_from_dict(data,list(cls.feature_names_in_))
    print(two_stage_predict(cls,reg,df))

if __name__=='__main__':
    run_cli()