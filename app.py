from flask import Flask, render_template, request
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/cnn', methods=['GET', 'POST'])
def show_cnn():
    class_fashion_label = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    if request.method == "POST":
        mode = request.form['mode']
        if mode == "Training":
            test_size = (100 - int(request.form['komposisi'])) / 100
            epochs = int(request.form['query_epoch'])
            #num_of_layers = int(request.form['query_layers'])
            tipeModel = request.form['tipeModel']
            
            mnist = fashion_mnist
            (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
            training_images = training_images.reshape(60000, 28, 28, 1)
            training_images = training_images / 255.0
            test_images = test_images.reshape(10000, 28, 28, 1)
            test_images = test_images / 255.0
            data_training, data_test, target_training, target_test = \
            train_test_split(training_images, training_labels, test_size=test_size, random_state=0)
            #if num_of_layers < 0 and num_of_layers > 10:
            #    num_of_layers = 5

            if tipeModel == "1":
                model = Sequential()
                model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
                model.add(Conv2D(32, kernel_size=3, activation='relu'))
        
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(10, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                print("training model 1")
                model.fit(training_images, training_labels, epochs=epochs)
                model.summary()
            elif tipeModel == "2":
                model = Sequential()
                model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(10, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                print("training model 2")
                model.fit(training_images, training_labels, epochs=epochs)
                model.summary()
            else:
                model = Sequential()
                model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(10, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                print("training model 3")
                model.fit(training_images, training_labels, epochs=epochs)
                model.summary()
            
            model.save('static/assets/model/cnn/model.h5') 
            test_loss, test_accuracy = model.evaluate(test_images, test_labels)

            return render_template('cnn.html', test_loss=test_loss, test_accuracy=test_accuracy)
        else:
            test = int(request.form['test_ke'])

            if test < 0 and test > 10000:
                test = 1
            mnist = fashion_mnist
            (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

            test_images_r = test_images.reshape(10000, 28, 28, 1)
            test_images_r = test_images_r / 255.0

            model = load_model('static/assets/model/cnn/model.h5')
            
            result = model.predict(test_images_r)
            print(result[0])
            kelasInd = 0;
            for i in range(len(result[test])):
                if result[test][i] > result[test][kelasInd]:
                    kelasInd = i
            kelas = class_fashion_label[kelasInd]
            
            plt.imshow(test_images[test],cmap='gray')
            plt.savefig('static/assets/images/hasil/hasil.png')
            print(kelas)
            return render_template('cnn.html', kelas=kelas)
    else:
        return render_template('cnn.html')

if __name__ == "__main__":
    app.run(threaded=False)
