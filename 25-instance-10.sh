#!/bin/bash

#dataPath, outputPath(for model, images, etc), trainInstances, validationInstances,testInstances,cropSize,learningRate,weightDecay,batchSize,nIter,epochs,trainModel,instance,useMinibatch,useValidation,blocks

#10
python /media/tensorflow/coffee/applications/subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 7_5,8_7,9_6,8_5,8_6,9_5 7_7 9_7,7_7,7_6 25 0.001 0.005 250 0 10 1 10 0 0 3 0 1
