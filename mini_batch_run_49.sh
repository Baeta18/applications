#!/bin/bash

#dataPath, outputPath(for model, images, etc), trainInstances, validationInstances,testInstances,cropSize,learningRate,weightDecay,batchSize,nIter,epochs,trainModel,instance,useMinibatch,useValidation,blocks
#ate 2 no 41
#1
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_5,9_6,7_7,7_6,7_5,8_7 7_7 8_6,9_7,9_5 49 0.001 0.005 250 0 3 1 1 0 0 4
#2
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_7,9_7,7_7,9_6,8_6,8_5 7_7 7_6,7_5,9_5 49 0.001 0.005 250 0 3 0 2 0 0 4
#3
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_7,9_6,7_5,8_5,9_5,7_6 7_7 9_7,7_7,8_6 49 0.001 0.005 250 0 3 1 3 0 0 4
#4
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 9_6,9_7,7_5,7_7,8_5,7_6 7_7 9_5,8_6,8_7 49 0.001 0.005 250 0 3 1 4 0 0 4
#5
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_7,8_5,9_5,7_5,7_7,8_6 7_7 7_6,9_6,9_7 49 0.001 0.005 250 0 3 1 5 0 0 4
#6
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 7_5,8_7,9_5,8_5,9_6,7_7 7_7 8_6,7_6,9_7 49 0.001 0.005 250 0 3 1 6 0 0 4
#7
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_6,8_7,9_7,8_5,7_6,7_5 7_7 9_5,7_7,9_6 49 0.001 0.005 250 0 3 1 7 0 0 4
#8
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 9_7,7_7,8_5,7_6,7_5,9_5 7_7 8_6,8_7,9_6 49 0.001 0.005 250 0 3 1 8 0 0 4
#9
#python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 8_7,9_5,8_5,7_5,7_7,9_7 7_7 8_6,9_6,7_6 49 0.001 0.005 250 0 3 1 9 0 0 4
#10
python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 7_5,8_7,9_6,8_5,8_6,9_5 7_7 9_7,7_7,7_6 49 0.001 0.005 250 0 3 1 10 0 0 4

