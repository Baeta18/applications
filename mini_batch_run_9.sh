#!/bin/bash

#dataPath, outputPath(for model, images, etc), trainInstances, validationInstances,testInstances,cropSize,learningRate,weightDecay,batchSize,nIter,epochs,trainModel,instance,useMinibatch,useValidation,blocks

python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 9_6,9_7,7_5,7_7,8_5,7_6 7_7 9_5,8_6,8_7 41 0.001 0.005 100 0 5 1 4 0 0 4

python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 9_6,9_7,7_5,7_7,8_5,7_6 7_7 9_5,8_6,8_7 41 0.001 0.005 100 0 5 1 4 0 0 3
