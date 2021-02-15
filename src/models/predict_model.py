import numpy as np
import pandas as pd
from joblib import dump

def model_submission(model, test_data, modelname='model_test', submissionname='submission_test'):
    """Perform a number of data processing
      - predict on test data 
      - save model
      - produce submission file

    Parameters
    ----------
    model: submission model
        the model to be saved and used for submission prediction
    test_data: pandas dataframe
        submission test data used for prediction
    modelname: text string 
        name of the model to be saved as (default: model_test)
    submissionname: text string
        name of the submission file (default: submission_test)

    Output
    -------
    saved model
    saved submission file
    
    Returns
    -------
    """
    
    modelname = 'models/' + modelname + '.joblib'
    submissionname = 'data/submission/' + submissionname + '.csv'
    dump(model, modelname)

    submission = model.predict_proba(test_data)[:,1]
    submission = pd.DataFrame({'TARGET_5Yrs': submission}).reset_index()
    submission['Id'] = submission.index
    submission = submission[['Id', 'TARGET_5Yrs']]
    submission.to_csv(submissionname, index=False)