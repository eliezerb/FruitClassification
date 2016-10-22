import os
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from toolbox import *

def get_features(filePath):
    
    imageBGR = cv2.imread(filePath)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    # Extracting fruit ROI
    segmentation, contour = segment_fruit(imageRGB)

    # Compute Features
    mean_color = get_dominant_color(imageRGB, segmentation)
    aspect_ratio = get_fruit_struct(imageRGB, contour)

    # Concatenate features in a vector
    x = [mean_color[0], mean_color[1], mean_color[2], aspect_ratio]

    return x

def load_training_set(classA, classB):
    X = []
    y = []

    # Read data from the first class
    files = os.listdir(classA)
    for i in files:
        filePath = classA + '/' + i
        x = get_features(filePath)
        
        y.append(0)
        X.append(x)

    # Read data from the second class
    files = os.listdir(classB)
    for i in files:
        
        filePath = classB + '/' + i
        x = get_features(filePath)

        y.append(1)
        X.append(x)

    return np.array(X), np.array(y)
    
    
if __name__ == '__main__':

    classA = './orange'
    classB = './apple'
    
    X, y = load_training_set(classA, classB)
    
    X_train = X[0:20:2, :];
    y_train = y[0:20:2];

    X_test = X[1:20:2, :];
    y_test = y[1:20:2];

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    print "Prediction:"
    print model.predict(X_test)
    print "\n"
    print "Expected Labels:"
    print y_test
