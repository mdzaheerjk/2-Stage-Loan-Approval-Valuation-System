import pandas as pd

def build_applicant_form_dict(d,expeted_col):
    df=pd.DataFrame(d)

    for c in df.select_dtypes(include=['object']).columns:
        df[c]=df[c].str.strip()

    missing=[c for c in expeted_col if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Input Colums : {missing}")
    return df[expeted_col]