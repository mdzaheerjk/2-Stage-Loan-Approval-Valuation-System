def two_stage_predict(cls,reg,applicant_df):

    proba=cls.predict_proba(applicant_df)
    preds=cls.predict(applicant_df)

    results=[]
    i=0
    approved_idx=1

    approved=int(preds[i])
    approved_prob=float(proba[i,approved_idx])


    reg_pred=None
    if approved==1:
        applicant_df_reg=applicant_df.copy()
        applicant_df_reg['loan_status']='Approve'
        reg_pred=float(reg.predict(applicant_df_reg)[0])
    results.append({
        "approved":approved,
        "approved_proba":approved_prob,
        "reg_pred":reg_pred
    })
    return results