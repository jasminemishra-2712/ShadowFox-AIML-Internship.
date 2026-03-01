import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("loan_data.csv")

print("Dataset preview:")
print(data.head())

# convert categorical to numeric
data = pd.get_dummies(data, drop_first=True)

# features and target
X = data.drop("Loan_Status_Y", axis=1)
y = data["Loan_Status_Y"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# evaluate
print("Accuracy:", accuracy_score(y_test, pred))
