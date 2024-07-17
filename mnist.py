import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.activations import softmax

#loading dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train) , (X_test , y_test) = mnist.load_data()

# pre-processing data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)


# defining  model
model= tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape= (28,28)))
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dense(10, activation ='softmax'))

# compling the model
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

# training the model
model.fit(X_train , y_train , epochs = 3) 

#calculating accuracy
loss,accuracy = model.evaluate(X_test, y_test)
print(accuracy)
print(loss)

#predicting on custom images
for x in range(0,10):

    img_path = (f"{x}.png")
    
    img = cv.imread(img_path , cv.IMREAD_GRAYSCALE)                # read and extract the image
    if img is None:
        print(f"error : could not read image{img_path}")
        continue
    img = cv.resize(img, (28, 28))                                 # Resize the image to 28x28
    img = np.expand_dims(img, axis=0)                              # Add batch dimension
    img = tf.keras.utils.normalize(img, axis=1)                    # Normalize pixel values

    prediction = model.predict(img)                                  #make predictions
    predicted_value = np.argmax(prediction)                        # get the predicted digit

    # print and display the predictions
    print("----------------")
    print("The predicted value is : ", np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0],cmap=plt.cm.binary)                          # display the image in gray scale
    plt.show()



