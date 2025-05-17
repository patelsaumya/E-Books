import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib

pd.options.mode.chained_assignment = None

df = pd.read_csv("house_data.csv")

X = df[["sq_feet", "num_bedrooms", "num_bathrooms"]]
y = df[["sale_price"]]

X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

X[X.columns] = X_scaler.fit_transform(X[X.columns])
y[y.columns] = y_scaler.fit_transform(y[y.columns])

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(
    loss='mean_squared_error',
    optimizer='SGD'
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    shuffle=True,
    verbose=2
)

joblib.dump(X_scaler, "X_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

model.save("house_value_model.h5")

print("Model training results:")

predictions_train = model.predict(X_train, verbose=0)

mse_train = mean_absolute_error(
    y_scaler.inverse_transform(predictions_train),
    y_scaler.inverse_transform(y_train)
)
print(f" - Training Set Error: {mse_train}")

predictions_test = model.predict(X_test, verbose=0)

mse_test = mean_absolute_error(
    y_scaler.inverse_transform(predictions_test),
    y_scaler.inverse_transform(y_test)
)
print(f" - Test Set Error: {mse_test}")
