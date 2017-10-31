#!/bin/bash

#dataPath, outputPath(for model, images, etc), trainInstances, validationInstances,testInstances,cropSize,learningRate,weightDecay,batchSize,nIter,epochs,trainModel,instance,useMinibatch,useValidation,blocks
#ate 2 no 41
#1
python /media/tensorflow/applications/subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_5,9_6,7_7,7_6,7_5,8_7 7_7 8_6,9_7,9_5 17 0.001 0.005 300 0 10 1 1 0 0 3 0 0.32
#2
python /media/tensorflow/applications/subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_7,9_7,7_7,9_6,8_6,8_5 7_7 7_6,7_5,9_5 17 0.001 0.005 300 0 10 1 2 0 0 3 0 0.32
#3
python /media/tensorflow/applications/subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_7,9_6,7_5,8_5,9_5,7_6 7_7 9_7,7_7,8_6 17 0.001 0.005 300 0 10 1 3 0 0 3 0 0.32
#4
python /media/tensorflow/applications/subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 9_6,9_7,7_5,7_7,8_5,7_6 7_7 9_5,8_6,8_7 17 0.001 0.005 300 0 10 1 4 0 0 3 0 0.32


