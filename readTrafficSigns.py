print('loading dependencies...')
import cv2
import csv
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

def readTrafficSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
        #print(gtFile)
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        counter = 0
        for row in gtReader:
            if counter == 25:
                break
            img = cv2.resize(cv2.imread(prefix + row[0]), (32,32))
            images.append(img)
            labels.append(row[7]) # the 8th column is the label
            counter += 1
        gtFile.close()
    return images, labels

print('loading data...')
data, target = readTrafficSigns('GTSRB/Final_Training/Images')
data, target = np.array(data), np.array(target, dtype=int)

randomize = np.arange(len(data))
np.random.shuffle(randomize)

data, target = data[randomize], target[randomize]
numOfTraining = int(len(data) * 80 / 100)
training_data, training_target, test_data, test_target = data[:numOfTraining], target[:numOfTraining], data[numOfTraining:], target[numOfTraining:]

print('create model...')
model = Sequential()

model.add( Conv2D(16, (3,3), input_shape=data[0].shape) )
model.add( Activation('relu') )

model.add( Conv2D(16, (3,3)) )
model.add( Activation('relu') )

model.add( MaxPooling2D(pool_size=(2,2)) )

model.add( Conv2D(32, (3,3)) )
model.add( Activation('relu') )

model.add( MaxPooling2D(pool_size=(2,2)) )

model.add( Flatten() )

model.add( Dense(256) )
model.add( Activation('relu') )
model.add( Dense(128) )
model.add( Activation('relu') )
model.add( Dense(43) )
model.add( Activation('softmax') )

model.summary()

#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('train model...')
model.fit(training_data, to_categorical(training_target), batch_size=32, epochs=10, validation_split=0.3, verbose=2)

print('testing model...')
print('testing data :', len(test_data))
predict = model.predict_classes(test_data)
predictTrue = np.sum(predict == test_target)
print('testing data true :', predictTrue)
acc = predictTrue / len(test_data)
print("testing accuracy :", acc)

cv2.imshow('image', test_data[0])
cv2.imshow('image', test_data[1])

loop = True
while loop:
    _input = input('prediksi data ke : ')
    if _input != 'exit':
        index = int(_input)
        predict = model.predict_classes([[data[index]]])
        print('predict :',predict)
        print('target  :',target[index])
    else:
        loop = False
