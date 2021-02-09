# DTS

This is the TensorFlow implementation of KDD submission.

## Dataset

### Individual
- Download data and save to (*dataset/ECG/*), which is named ECG5000 on the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/).

### Group
- Download data and convert to .tfrecord files for TensorFlow
  (*./generate_tfrecords.sh*)

## Overview

### Individual
- Download data and save to (*dataset/ECG/*), which is named ECG5000 on the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/).
- Train and Test models (*individual/ts.py*)


### Group
- Download data and convert to .tfrecord files for TensorFlow
  (*./generate_tfrecords.sh*)
- Train models (*group/main.py*)
- Evaluate models (*group/main_eval.py* )
- Analyze results (*group/analysis.py*)


### Individual
Train a DTS model (i.e. using DTS) on ECG.

    python3 ts.py
    

### Group
Train a DTS model (i.e. using DTS) on person 14 of the UCI HAR dataset
and adapt to person 19.

    python3 main.py \
        --logdir=example-logs --modeldir=example-models \
        --method=yndaws --dataset=ucihar --sources=14 \
        --target=19 --uid=0 --debugnum=0 --gpumem=0

