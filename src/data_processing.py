import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_data():
   df = pd.read_csv("data/fraud_data.csv")
   return df
def preprocess_data(df):
   X = df[['feature1', 'feature2']]
   y = df['is_fraud']
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
