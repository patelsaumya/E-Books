import pandas as pd
from tensorflow.keras.models import load_model
import joblib

model = load_model('house_value_model.h5')

X_scaler = joblib.load('X_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

house_1 = [
    2000,
    3,
    2
]

homes = pd.DataFrame([
    house_1
], columns=['sq_feet', 'num_bedrooms', 'num_bathrooms'])

scaled_home_data = X_scaler.transform(homes)

home_values = model.predict(scaled_home_data)

unscaled_home_values = y_scaler.inverse_transform(home_values)

predicted_value = unscaled_home_values[0][0]

print("House details:")
print(f"- {house_1[0]} sq feet")
print(f"- {house_1[1]} bedrooms")
print(f"- {house_1[2]} bathrooms")
print(f"Estimated value: ${predicted_value:,.2f}")

