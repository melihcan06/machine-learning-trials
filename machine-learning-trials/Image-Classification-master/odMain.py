import keras
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import cifar10

epochs = 100
batch_size = 32
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))

model.save("cifar10_model")
#model=keras.models.load_model("cifar10_model")

img=cv2.imread("kedi.jpg")

img=cv2.resize(img,(32,32))
img=np.reshape(img,(-1,32,32,3))

img=img.astype('float32')
img/=255

print(model.predict(img))
print(model.predict_classes(img))
model.summary()
