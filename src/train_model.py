from sklearn.ensemble import RandomForestClassifier
from data_processing import load_data, preprocess_data
from evaluate_model import evaluate
def train():
   df = load_data()
   X_train, X_test, y_train, y_test = preprocess_data(df)
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   preds = model.predict(X_test)
   evaluate(y_test, preds)
if __name__ == "__main__":
   train()
