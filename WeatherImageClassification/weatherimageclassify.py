# main.py

# Original Date: May 2022
# Modify Date: Aug 2025
# Juni Learning Starter Code:
# https://colab.research.google.com/drive/1MAlqUFFcjaLPQJXvUMHBjwx_igwkemPY?usp=sharing

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import matplotlib.pyplot as plt

imagegen = ImageDataGenerator(rescale=1./255., rotation_range=30, horizontal_flip=True, validation_split=0.1)

PATH = os.path.join(os.path.dirname('weatherData'), 'weatherData')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

train_generator = imagegen.flow_from_directory(train_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
validation_generator = imagegen.flow_from_directory(validation_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
test_generator = imagegen.flow_from_directory(test_dir, class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4,activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# predictions=model.fit_generator(train_generator,validation_data=validation_generator,epochs=10)
earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
predictions = model.fit(train_generator, validation_data=validation_generator, epochs=30, callbacks=[earlystop])
# predictions = model.fit(train_generator, validation_data=validation_generator, epochs=10)

eval=model.evaluate(test_generator)[1]
print(eval)

examples_datagen = ImageDataGenerator(rescale=1./255)
examples_generator = examples_datagen.flow_from_directory(PATH, target_size=(224, 224), classes=['examples'], shuffle=False)
examples_generator.reset()

predict = model.predict(examples_generator)
output=[]
for image in predict:
  max = 0
  index = 0
  for i in range(len(image)):
    if image[i] > max:
      max = (image[i])
      index = i
  output.append(index)
print(output)
#[0, 1, 3, 2]

for i in range(len(examples_generator.filenames)):
  print('Filename: ' + examples_generator.filenames[i])
  print('Label: ' + list(train_generator.class_indices.items())[i][0])
  print()
#[0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3]
#[cloudy,rainy,sunny,sunrise]

'''model.save('/model save')
model = keras.models.load_model('path/to/location')'''

# next = test_generator.next()
next = next(test_generator)
plt.imshow(next[0][20])
plt.axis('off')
plt.show()