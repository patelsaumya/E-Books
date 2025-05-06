import joblib
import pandas as pd

model = joblib.load('house_value_model.pkl')

house_1 = [
    2000,
    3,
    2
]

homes = pd.DataFrame([
    house_1
], columns=['sq_feet', 'num_bedrooms', 'num_bathrooms'])

home_values = model.predict(homes)

predicted_value = home_values[0]

print("House details:")
print(f"- {house_1[0]} sq feet")
print(f"- {house_1[1]} bedrooms")
print(f"- {house_1[2]} bathrooms")
print(f"Estimated value: ${predicted_value:,.2f}")
