from console_utils import *

# Modeli kaydetme
joblib.dump(voting_clf, 'voting_model(lgbm_xgb_rf).pkl')