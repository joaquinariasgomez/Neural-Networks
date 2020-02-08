import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

bd = keras.datasets.fashion_mnist #Fashion MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = bd.load_data()

# Split train set into train and validation set, and also normalize it so that values are in between 0.0 and 1.0 (both floats)
X_train, X_valid = X_train_full[5000:] / 255.0, X_train_full[:5000] / 255.0
y_train, y_valid = y_train_full[5000:], y_train_full[:5000]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Time to build neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax")) #Softmax because the classes are exclusive
#print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",    #This means we will train the model using simple Stochastic Gradient Descent (backpropagation algorithm)
                                    #To tune the learing rate (which defaults to 0.01), use: optimizer=keras.optimizer.SGD(lr=???)
                metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
#Print results
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)    #Vertical range to [0-1]
plt.show()