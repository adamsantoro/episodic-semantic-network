from __future__ import division
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activatePolicy(policy, tau):
    return np.clip(policy * np.exp(-1 / tau) , 0, 1)

def convertToProbability(x, NewMin, NewMax):
    OldMin = np.amin(x)
    OldMax = np.amax(x)
    new_x = (((x - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return new_x / np.sum(new_x)
