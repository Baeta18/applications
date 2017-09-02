#!/bin/bash

#dataPath, outputPath(for model, images, etc), trainInstances, validationInstances,testInstances,cropSize,learningRate,weightDecay,batchSize,nIter,epochs,trainModel,useMinibatch,useValidation,blocks,gpuUse

python /media/tensorflow/coffee/applications/mini_batch_subimages_6x3.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ train validation teste 65 0.001 0.005 100 0 10 1 0 5 1

