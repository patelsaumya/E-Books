import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("house_data.csv")

X = df[["sq_feet", "num_bedrooms", "num_bathrooms"]]
y = df["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, 'house_value_model.pkl')

print("Model training results:")

mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Test Set Error: {mse_test}")