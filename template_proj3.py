from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np

# Model Template

model = Sequential() # declare model
model.add(Dense(60, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
# model here
model.add(Dense(100, activation="relu"))
# model.add(Dense(100, activation="relu"))
# model.add(Dense(100, activation="relu"))
# model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="selu"))
# 
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Load Data
images = np.load('images.npy')
labels = np.load('labels.npy')

# Split Data
mask = np.random.rand(len(images)) <= .6
x_train = images[mask]
y_train = labels[mask]
y_train = to_categorical(y_train, 10)

remaining_images = images[~mask]
remaining_labels = labels[~mask]

mask = np.random.rand(len(remaining_images)) <=  .375
x_val = remaining_images[mask]
y_val = remaining_labels[mask]
y_val = to_categorical(y_val, 10)

x_test = remaining_images[~mask]
y_test = remaining_labels[~mask]
y_test = to_categorical(y_val, 10)

# Normalize Data
x_val, x_train = x_val / 255, x_train / 255
x_test = x_test / 255

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=20,
                    batch_size=512)


# Report Results

print(history.history)
model.predict(x_test)
#model.summary()