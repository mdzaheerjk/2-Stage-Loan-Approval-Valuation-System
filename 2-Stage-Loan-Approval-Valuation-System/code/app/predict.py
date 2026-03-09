def two_stage_predict(cls,reg,appliacnat_df):
    proba=cls.predict_prob(appliacnat_df)
    preds=cls.predict(appliacnat_df)

    results=[]
    i=0
    approved_idx=1

    approved=int(preds[i])
    approved_prob=float(proba[1,approved_idx])


    reg_pred=None
    if approved==1:
        reg_appliacnat_df=appliacnat_df.copy()
        reg_appliacnat_df['loan_status']='Approve'
        reg_pred=float(reg.predict(reg_appliacnat_df)[0])
    results.append({
        'approved':approved,
        'approved_prob':approved_prob,
        'reg_pred':reg_pred
    })

    return results