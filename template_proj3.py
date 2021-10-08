from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))

# Fill in Model Here
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Load Data
images = np.load('images.npy')
labels = np.load('labels.npy')

# Split Data
mask = np.random.rand(len(images)) <= .6
x_train = images[mask]
y_train = labels[mask]

remaining_images = images[~mask]
remaining_labels = labels[~mask]

mask = np.random.rand(len(remaining_images)) <=  .375
x_val = remaining_images[mask]
y_val = remaining_labels[mask]

x_test = remaining_images[~mask]
y_test = remaining_labels[~mask]

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=10, 
                    batch_size=512)


# Report Results

print(history.history)
model.predict()

