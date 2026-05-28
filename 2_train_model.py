import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("🔄 Loading dataset from 'sign_data.csv'...")

# 1. Load the data from the CSV file
# Each row has 42 coordinates + 1 text label
data = pd.read_csv('sign_data.csv', header=None)

# 2. Split the data into features (X) and labels (y)
X = data.iloc[:, :-1].values  # All columns except the last one (the 42 coordinates)
y = data.iloc[:, -1].values   # The very last column (the text sign labels)

# 3. Split dataset into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Dataset Loaded! Total samples: {len(X)}")
print(f"💪 Training set size: {len(X_train)} samples")
print(f"🧪 Testing set size: {len(X_test)} samples")

# 4. Initialize and train the Random Forest Classifier
print("\n🤖 Training the Random Forest Machine Learning Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model's accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Complete!")
print(f"🎯 Model Accuracy Score: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file using pickle
model_filename = 'sign_language_model.p'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Trained model successfully saved as '{model_filename}'!")