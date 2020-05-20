import tensorflow as tf
import cv2
import numpy as np

model=tf.keras.models.load_model('mnist_classifier.h5')

#reading the image
img=cv2.imread(r'C:\Users\yusuf\Downloads\my projects\Digit_recogniser\custom test\hand.jpg')

# changing the color channel
img_predict=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_predict=cv2.resize(img_predict,(28,28))
img_predict=np.reshape(img_predict,(1,28,28,1))
print(img_predict.shape)

print(np.argmax(model.predict(img_predict)))

cv2.imshow('image',img)
cv2.waitKey(0)
