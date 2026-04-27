from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
def load_data():
   # dummy example (we’ll replace later)
   data = pd.DataFrame({
       'feature1': [1,2,3,4,5,6],
       'feature2': [2,3,4,5,6,7],
       'is_fraud': [0,0,0,1,1,1]
   })
   return data
def train():
   df = load_data()
   X = df[['feature1','feature2']]
   y = df['is_fraud']
   model = RandomForestClassifier()
   model.fit(X, y)
   preds = model.predict(X)
   print(classification_report(y, preds))
if __name__ == "__main__":
   train()
