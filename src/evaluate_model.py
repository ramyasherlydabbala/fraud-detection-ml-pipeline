from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def evaluate(y_true, y_pred):
   print("Accuracy:", round(accuracy_score(y_true, y_pred), 2))
   print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 2))
   print("Recall:", round(recall_score(y_true, y_pred, zero_division=0), 2))
   print("F1 Score:", round(f1_score(y_true, y_pred, zero_division=0), 2))
   print("Confusion Matrix:")
   print(confusion_matrix(y_true, y_pred))
