import json
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data  
y = iris.target  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


cv_scores = cross_val_score(model, X_scaled, y, cv=5)
cv_mean = cv_scores.mean()


metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cv_mean": cv_mean,
    "classification_report": classification_report(y_test, y_pred, target_names=iris.target_names)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)


joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo entrenado y guardado. Métricas:")
print(metrics)