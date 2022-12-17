from libraryLearning import *
import numpy as np

numberOfEdgeDevices = 1
numberOfCommunicationRound = 1000
numberOfImagesPerLabel = 50
accuracyCalculationRate = 5 # communications round
isHeterogeneous = False

learningEngines = [];
for indED in range(numberOfEdgeDevices):
    if isHeterogeneous == True:
        parameters = dict([
            ('labels', np.array([0, 1, 2, 3, 4]+np.array([indED]))), 
            ('indexShard', indED), 
            ('numberOfImagesPerLabel', numberOfImagesPerLabel), 
            ])
    else:
        parameters = dict([
            ('labels', np.arange(10)), 
            ('indexShard', indED), 
            ('numberOfImagesPerLabel', numberOfImagesPerLabel), 
            ])
    imagesTraining, labelsTraining = fcn_dataLoader(parameters)


    parameters = dict([
        ('batchSizeTrain', 100), 
        ('learningRate', 1e-4), 
        ('imagesTraining', imagesTraining), 
        ('labelsTraining', labelsTraining), 
        ])
    learningEngines.append(objLearningEngine(parameters))


acc = np.zeros((numberOfEdgeDevices,numberOfCommunicationRound))
majorityVote = np.zeros((learningEngines[0].numberOfParameters))

for indComm in range(numberOfCommunicationRound):
    gradientsAccumulated = np.zeros((learningEngines[0].numberOfParameters))
    for indED in range(numberOfEdgeDevices):
        gradientsAccumulated += np.sign(learningEngines[indED].step(majorityVote))
        print ('ED' + str(indED) + f', Loss: {learningEngines[indED].lossChange[indComm]:.4f}')
    majorityVote = np.sign(gradientsAccumulated)


    if (indComm) % accuracyCalculationRate == 0:
        for indED in range(numberOfEdgeDevices):
            acc[indED,indComm] = fcn_calculateTestAccuracy(learningEngines[indED].model, learningEngines[indED].device)
            print ('Round: '+ str(indComm) + ', ED' + str(indED) + f', Accuracy: {acc[indED,indComm]:.4f}')