# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/7/18 6:59 PM
# project   DeepLearingStudy

import numpy as np
import argparse
import os
import random
import time
import collections

batchSize = 64

learningRateBase = 0.001
learningRateDecayStep = 1000
learningRateDecayRate = 0.95

epochNum = 10                    # train epoch
generateNum = 5                   # number of generated poems per time

type = "poetrySong"                   # dataset to use, shijing, songci, etc
trainPoems = "/mnt/hgfs/WorkSpace/data/poem_generator/dataset/" + type + "/" + type + ".txt" # training file location
checkpointsPath = "/mnt/hgfs/WorkSpace/data/poem_generator/checkpoints/" + type # checkpoints location

saveStep = 1000                   # save model every savestep



# evaluate
trainRatio = 0.8                    # train percentage
evaluateCheckpointsPath = "/mnt/hgfs/WorkSpace/data/poem_generator/checkpoints/evaluate"