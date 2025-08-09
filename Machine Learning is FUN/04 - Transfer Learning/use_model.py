from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import xception
image_to_test = "bird_1.png"

model = load_model('bird_feature_classifier_model.h5')

img = image.load_img(image_to_test, target_size=(73, 73))

image_array = image.img_to_array(img)

images = np.expand_dims(image_array, axis=0)

images = xception.preprocess_input(images)

feature_extraction_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))
features = feature_extraction_model.predict(images)

results = model.predict(features)

single_result = results[0][0]

print(f"Likelihood that {image_to_test} is a bird: {single_result * 100}%")