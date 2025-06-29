from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
import numpy as np

model = load_model('bird_model.h5')

for f in sorted(Path(".").glob("*.png")):

    image_to_test = image.load_img(str(f), target_size=(32, 32))

    image_to_test = image.img_to_array(image_to_test)

    image_to_test /= 255

    list_of_images = np.expand_dims(image_to_test, axis=0)

    results = model.predict(list_of_images)

    image_likelihood = results[0][0]

    if image_likelihood > 0.5:
        print(f"{f} is most likely a bird! ({image_likelihood:.2f})")
    else:
        print(f"{f} is most likely NOT a bird! ({image_likelihood:.2f})")
