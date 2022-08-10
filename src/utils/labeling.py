import numpy as np 

class LabelGenerator:

    def __init__(self, params):

        self.composite = params.get("composite", False)  
        self.composite_labels = params.get("composite_labels", {}) 
    
    def fit(self, trainLabels, testLabels):
        if self.composite:
            print("using the following composite labels: " + str(self.composite_labels.keys()))
            trainY = np.zeros(trainLabels.shape)
            testY = np.zeros(testLabels.shape)

            for label in self.composite_labels:
                label_set = self.composite_labels[label]
                trainY[np.where(np.isin(trainLabels, label_set))] = label 
                testY[np.where(np.isin(testLabels, label_set))] = label 
        else: 
            trainY = trainLabels 
            testY = testLabels
        
        return trainY, testY
