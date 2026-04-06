def two_stage_predict(cls,reg,applicant_df):

    proba=cls.predict_proba(applicant_df)
    preds=cls.predict(applicant_df)

    results=[]
    i=0
    
    # Get the class indices dynamically
    classes = cls.classes_
    approved_idx = list(classes).index(1) if 1 in classes else 1
    rejected_idx = list(classes).index(0) if 0 in classes else 0
    
    approved=int(preds[i])
    approved_prob=float(proba[i,approved_idx])
    rejected_prob=float(proba[i,rejected_idx])
    
    # Debug: Print full probability distribution
    print(f"Debug - Classes: {classes}")
    print(f"Debug - All probabilities: {proba[i]}")
    print(f"Debug - Approved prob: {approved_prob}, Rejected prob: {rejected_prob}")

    reg_pred=None
    if approved==1:
        applicant_df_reg=applicant_df.copy()
        applicant_df_reg['loan_status']='Approve'
        reg_pred=float(reg.predict(applicant_df_reg)[0])
    results.append({
        "approved":approved,
        "approved_proba":approved_prob,
        "rejected_proba":rejected_prob,
        "reg_pred":reg_pred
    })
    return results