#!/bin/bash
#'dataPath', 'outputPath(for model, images, etc)','instance' , 'fusion-instances','cropSize','fusion-type'
#python /media/tensorflow/coffee/applications/maps-fusion.py /media/tensorflow/coffee/output/results/probability/ /media/tensorflow/coffee/output/results/maps-fusion/ 2 7_6,9_6,9_7 25,33,41 1
python maps-fusion.py /home/baeta/Documentos/patreo/applications/ /home/baeta/Documentos/patreo/applications/ 3 8_6 25,33,41 1
