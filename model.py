import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import json

# Load & clean data
data = pd.read_csv("HistoricalQuotes.csv")
data = data.dropna()

# Remove spaces from column names
data.columns = data.columns.str.strip()

# Remove $ and convert to float
for col in ['Open', 'High', 'Low', 'Close/Last']:
    data[col] = data[col].str.replace('$', '', regex=False).astype(float)

# Features & target
X = data.drop(columns=['Date', 'Close/Last'])
y = data['Close/Last']

# Save the order of feature names
feature_names = list(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Save model and features
joblib.dump(rf, 'model.pkl')
joblib.dump(feature_names, 'features.pkl')

# Accuracy
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Save metrics
metrics = {'accuracy': r2}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=1)

print(" Model and features saved successfully!")
