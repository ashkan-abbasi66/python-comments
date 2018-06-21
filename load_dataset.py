from __future__ import print_function
from scipy import misc
import os
import numpy as np
import sys

# Contents:
# prepare_data(task)
# load_batch(task, TRAIN_SIZE, IMAGE_SIZE)


# Suitable for a image to image translation task.
def prepare_data(task):
    # INPUTS:
    #   task: a folder name
    #   We assume, the folder structure is something like this:
    #   training dataset:
    #       data/training/inputs/   --- directory containing the input images
    #       data/training/task/     --- desired outputs for a given task
    #   test dataset:
    #       data/test/inputs/
    # OUTPUTS:
    # Three list containing the names of input/output pairs and test data:
    #   input_names,output_names,val_names

    # training dataset
    input_names=[]
    output_names=[]
    N_train = 10  # number of training images
    for i in range(1,N_train+1):
        input_names.append("../data/training/inputs/%06d.png"%(i))# 000001.png
        output_names.append("../data/training/%s/%06d.png"%(task,i))

    # test/evaluation dataset
    N_test=18
    val_names = []
    for i in range(1,N_test+1):
        val_names.append("../data/test/inputs/%06d.png"%(i))

    return input_names,output_names,val_names

# Suitable for a patch-based training strategy
def load_batch(task, TRAIN_SIZE, IMAGE_SIZE):
    # Loads a random subset of training dataset or the whole dataset (TRAIN_SIZE=-1)
    #
    # INPUTS:
    #   task: a folder name
    #   We assume, the folder structure is something like this:
    #   training dataset:
    #       data/training/inputs/   --- directory containing the input patches
    #       data/training/task/     --- desired output patches for a given task
    #   test dataset:
    #       data/test/inputs/
    #
    #   TRAIN_SIZE: subset size
    # OUTPUTS:
    #   train_data: a matrix with N by d size. (N: # of samples which is determined by TRAIN_SIZE,
    #       d: dimension of data vector which is determined by IMAGE_SIZE)
    #   train_data contains input patches in vector form
    #   train_answ contains the output of the given task

    inputsDir ='data/training/inputs/' # contains input patches
    outputsDir = 'data/training/%s/'%task

    # os.listdir(inputsDir) is a list.
    # E.g., ['1.jpg','2.jpg',...]
    # for i in os.listdir(inputsDir):
    #   print(i)
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(inputsDir)
                               if os.path.isfile(os.path.join(inputsDir, name))])

    # if TRAIN_SIZE == -1 then load all training patches
    if TRAIN_SIZE == -1:
        TRAIN_SIZE = NUM_TRAINING_IMAGES
        TRAIN_IMAGES = np.arange(0, TRAIN_SIZE) # indices of the training set
    else:
        # select a subset of indices without repetition
        TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, IMAGE_SIZE))
    train_answ = np.zeros((TRAIN_SIZE, IMAGE_SIZE))

    i = 0
    for img in TRAIN_IMAGES:
        I = np.asarray(misc.imread(inputsDir + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_data[i, :] = I

        I = np.asarray(misc.imread(outputsDir + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_answ[i, :] = I

        i += 1
        if i % 100 == 0:
            print(str(round(i * 100 / TRAIN_SIZE)) + "% done", end="\r")

    return train_data, train_answ
