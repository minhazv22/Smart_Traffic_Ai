                                                   
  <!--[![Backers on Open Collective](https://opencollective.com/nest/backers/badge.svg)](https://opencollective.com/nest#backer)
  [![Sponsors on Open Collective](https://opencollective.com/nest/sponsors/badge.svg)](https://opencollective.com/nest#sponsor)-->
## Real-Time Traffic Detection And Management from an Image using Machine Learning Approach (Part of Smart Street:Street of Next Generation)

## Description

We introduce a traffic system that is more robust and minimize waiting time. The system intends to identify high traffic and low traffic in order to control traffic on each side of the road with the help of a machine learning model. In this paper, we use a transfer learning technique where Pre-trained CNN networks are loaded to extract features from the image. The system will take a video feed of the roads and feed it to a machine learning model. The overall system will consist of a server and a client-side. The machine learning model is made up of a VGG-16 network for the feature extraction and uses SVM to perform the classification.

<p>
<a href="https://drive.google.com/file/d/1HX2p9uDoUI-qqHLXU-eCepNC6aVT3q7g/view" target="_blank">Video</a>
</p>
<p>
<a href="https://drive.google.com/file/d/1vcpDvDnOvZFKNANMCl8xO_lEOCRvyk0M/view" target="_blank">Poster</a>
</p>
<p>
<a href="https://github.com/MunPotter/Traffic-AI/tree/main/server_example" target="_blank">Inference</a>
</p>

```bash
this is confidential project so i just provide some function.For inference check server example.

```
## Library
```bash
tensorflow
keras
numpy
matplotlib
sklearn
opencv-python
```


## Import library

```bash

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
```

## Feature extraction using VGG16

```bash
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features
```

## train and test data selection

```bash
  
from sklearn.model_selection import KFold
kf = KFold(n_splits=6)

kf.get_n_splits(x)
KFold(n_splits=6, random_state=None, shuffle=False)
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

```

## training model

```bash
  
#KNN classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
clfknn = KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(clfknn, x, y, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#svm classifier

from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='linear')
scoresvm = cross_val_score(clf, x, y, cv=4)
scoresvm
print("Accuracy: %0.2f (+/- %0.2f)" % (scoresvm.mean(), scoresvm.std() * 2))


#KNN classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
clfknn = KNeighborsClassifier(n_neighbors=5)
scores3 = cross_val_score(clfknn, x, y, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))



#random forest classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clfrf=RandomForestClassifier(n_estimators=150)
scores4 = cross_val_score(clfrf, x, y, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std() * 2))

```

## License
A product of Shazzadul Islam
