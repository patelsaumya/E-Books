from pathlib import Path
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception

x_train = []
y_train = []

for image_file in Path("training_dataset").glob("**/*.png"):

    image_data = image.load_img(image_file, target_size=(73, 73))

    image_array = image.img_to_array(image_data)

    x_train.append(image_array)

    if "not_bird" in image_file.stem:
        y_train.append(0)
    else:
        y_train.append(1)

x_train = np.array(x_train)

x_train = xception.preprocess_input(x_train)

feature_extractor = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))

features_x = feature_extractor.predict(x_train)

joblib.dump(features_x, "x_train.dat")

y_train = np.array(y_train)

joblib.dump(y_train, "y_train.dat")
