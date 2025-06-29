from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_test = y_test == 2

X_test = X_test.astype('float32')
X_test /= 255

model = load_model('bird_model.h5')
predictions = model.predict(X_test, batch_size=32, verbose=1)

predictions = predictions > 0.5

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

report = classification_report(y_test, predictions)
print(report)