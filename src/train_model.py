from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data_processing import load_data, preprocess_data
from evaluate_model import evaluate
def train_models():
   df = load_data("data/fraud_data.csv")
   X_train, X_test, y_train, y_test = preprocess_data(df)
   # Random Forest
   rf_model = RandomForestClassifier(random_state=42)
   rf_model.fit(X_train, y_train)
   rf_preds = rf_model.predict(X_test)
   print("Random Forest Results:")
   evaluate(y_test, rf_preds)
   print("\n---------------------\n")
   # XGBoost
   xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
   xgb_model.fit(X_train, y_train)
   xgb_preds = xgb_model.predict(X_test)
   print("XGBoost Results:")
   evaluate(y_test, xgb_preds)
if __name__ == "__main__":
   train_models()
