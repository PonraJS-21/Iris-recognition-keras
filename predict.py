import pickle
from keras.models import model_from_json
import numpy as np
import os
import cv2


classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()


# load json and create model
json_file = open('model/model_epochs_150.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model_epochs_150.h5")
print("Model is now loaded in the disk")

root = 'predict'
folders = os.listdir(root)
for folder in folders:
    print('Actual: {} \n{}'.format(folder, '------------'))
    tempDirectory = os.path.join(root, folder)
    for img in os.listdir(tempDirectory):
        image = np.array(cv2.imread(os.path.join(tempDirectory, img)))
        image = cv2.resize(image, (80, 360))
        image = np.array([image])
        image = image.astype('float32')
        image = image / 255.0

        probability = loaded_model.predict(image)

        confidence = np.max(probability)

        prediction = int_to_word_out[np.argmax(probability)]

        print('{} predicted as {} with {} confidence'.format(
            img, prediction, confidence))
