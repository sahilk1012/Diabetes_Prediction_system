import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# 1. Load the Dataset
# Ensure 'diabetes.csv' is in the same directory as this script
diabetes_data = pd.read_csv('diabetes.csv')

# 2. Separate Features and Target (Assuming target column is named 'Outcome')
X = diabetes_data.drop('Outcome', axis=1)
Y = diabetes_data['Outcome']

# 3. Data Standardization
# This is critical: we must save this scaler to use in the frontend
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# 4. Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# 5. Train the Model (Using Support Vector Machine here, but you can swap it)
classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# 6. Save the trained model AND the scaler as pickle files
pickle.dump(classifier, open('trained_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

print("Model and Scaler trained and saved successfully!")