
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

# scaling
x_train=x_train/255
x_test=x_test/255

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
# one hot encoding
from tensorflow.keras.utils import to_categorical

y_cat_train=to_categorical(y_train,10)
y_cat_test=to_categorical(y_test,10)

# building model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# adding early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss',patience=2)

# training the model

model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

summary=model.summary()
# evaluation

metrics=pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

# testing/predicting

from sklearn.metrics import classification_report,confusion_matrix

predictions=model.predict_classes(x_test)
report=classification_report(y_test,predictions)

model.save('mnist_classifier.h5')

