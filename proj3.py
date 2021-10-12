# CS 4341 Project 3
# Alex Hunt, Matthew Nagy, Matthew Vindigni
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Model Template

model = Sequential() # declare model
model.add(Dense(60, input_shape=(28*28, ), kernel_initializer='random_normal')) # first layer
model.add(Activation('relu'))
# Hidden Layers
model.add(Dense(100, activation="relu"))
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
print("labels")
print(labels[0])

# Split Data
mask = np.random.rand(len(images)) <= .6
x_train = images[mask]
y_train = labels[mask]
y_train = to_categorical(y_train, 10)
print(y_train[0])

remaining_images = images[~mask]
remaining_labels = labels[~mask]

mask = np.random.rand(len(remaining_images)) <=  .375
x_val = remaining_images[mask]
y_val = remaining_labels[mask]
y_val = to_categorical(y_val, 10)

x_test = remaining_images[~mask]
y_test = remaining_labels[~mask]
y_true = y_test
y_test = to_categorical(y_test, 10)

# Normalize Data values to 0-1 from 0-255
x_val, x_train = x_val / 255, x_train / 255
x_test = x_test / 255

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val),
                    epochs=20,
                    batch_size=128)


# Report Results

print(history.history)

predictions = model.predict(x_test)

# Save Model

model.save('best-model')

imgnames=['1','2','3']
# Visualize Grayscale Images of misclassified objects
def visualize(img, i):
    plt.imshow(np.reshape(img, (28,28)), cmap='gray', vmin=0, vmax=1)
    plt.savefig(imgnames[i])

# Produce Confusion Matrix
matrix = np.zeros((10,10), dtype=int)
imgs = 0
for i in range(len(predictions)):
    row = np.argmax(predictions[i])
    col = np.argmax(y_test[i])
    matrix[row][col] += 1
    if row != col and imgs < 3:
        visualize(x_test[i], imgs)
        imgs+=1


#model.summary()
