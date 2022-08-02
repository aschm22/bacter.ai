##########################################
##########  Bacter.ai   v1.041  ##########
##########    Copyright 2022    ##########
##########     Adam Schmidt     ##########
##########################################


from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt

def main():
    training_set = preprocessing.image_dataset_from_directory('C:/Users/amsch/Anaconda3/bacteria', 
                                                              validation_split=0.2, subset="training", 
                                                              label_mode="categorical",
                                                             seed=0, image_size=(400,400))
    test_set = preprocessing.image_dataset_from_directory('C:/Users/amsch/Anaconda3/bacteria', 
                                                          validation_split=0.2, subset="validation", 
                                                          label_mode="categorical",
                                                         seed=0, image_size=(400,400))
    
    
    
    # build CNN
    m = Sequential()
    m.add(Rescaling(1/255))
    m.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(400,400,3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(256, kernel_size=(3, 3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(512, kernel_size=(3, 3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(1024, kernel_size=(3, 3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(2, activation='softmax'))
    
    m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
    history = m.fit(training_set, batch_size=16, epochs=100)
    print(history.history['accuracy'])
    print(training_set.class_names)
    
    print("testing")
    score = m.evaluate(test_set)
    print('Test accuracy for main:', score[1])
    
    plt.plot(history.history["accuracy"])
    plt.title('Accuracy')
    plt.ylabel('%')
    plt.xlabel('Epochs')
    plt.show()
    
    image_file = "C:/Users/amsch/Anaconda3/bacteria_test/test_plate.jpg"
    img = preprocessing.image.load_img(image_file, target_size=(400,400))
    img_arr = preprocessing.image.img_to_array(img)
    
    plot = plt.imshow(img_arr.astype('uint8'))
    plt.show()
    
    img_cl = img_arr.reshape(1, 1000, 1000, 3)
    score = m.predict(img_cl)
    print(score)
    
    m.save("C:/Users/amsch/Anaconda3/functions/mainneuralnetwork.h5")
 
    return m
          
    
main()
