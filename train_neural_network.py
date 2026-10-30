import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the input (scale pixel values to the range [0, 1])
x_train = x_train/255.0
x_test = x_test/255.0

# one-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)), # 784 input features (28x28 pixels)
    Dense(128, activation='relu'), # 128 neurons
    Dense(10, activation='softmax') # 10 output classes (digits 0-9)
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# evaluate the model
_, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}",)
