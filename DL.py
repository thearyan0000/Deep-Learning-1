import tensorflow as tf
from keras import layers, models
from keras.datasets import cifar10
from keras.utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='linear', input_shape=(32, 32, 3), padding='same'))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(32, (3, 3), activation='linear', padding='same'))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='linear'))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc * 100:.2f}%')
