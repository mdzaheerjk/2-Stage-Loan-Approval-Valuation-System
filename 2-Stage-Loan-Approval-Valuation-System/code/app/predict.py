def two_stage_predict(cls,reg,appliacnat_df):
    proba=cls.predict_prob(appliacnat_df)
    preds=cls.predict(appliacnat_df)

    results=[]