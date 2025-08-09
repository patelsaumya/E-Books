import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


x_test = []
y_test = []

for image_file in Path("test_dataset").glob("**/*.png"):

    image_data = image.load_img(image_file, target_size=(73, 73))

    image_array = image.img_to_array(image_data)

    x_test.append(image_array)

    if "not_bird" in image_file.stem:
        y_test.append(0)
    else:
        y_test.append(1)

x_test = np.array(x_test)

x_test = xception.preprocess_input(x_test)

feature_extractor = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))

model = load_model("bird_feature_classifier_model.h5")

features_x = feature_extractor.predict(x_test)

predictions = model.predict(features_x)

predictions = predictions > 0.5

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

report = classification_report(y_test, predictions)
print(report)