import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# correct path to dataset inside Beginner folder
data = pd.read_csv("Beginner/Boston_House_Price_Prediction/HousingData.csv")

print("Dataset preview:")
print(data.head())

# handle missing values
data = data.fillna(data.mean())

# split features and target
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# results
print("\nBoston House Price Prediction Results")
print("MSE:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))
