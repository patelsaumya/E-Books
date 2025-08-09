from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import joblib

x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    validation_split=0.05,
    epochs=10,
    shuffle=True,
    verbose=2
)

model.save("bird_feature_classifier_model.h5")