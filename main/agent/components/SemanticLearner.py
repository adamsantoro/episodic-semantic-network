from __future__ import division
import numpy as np
from ...lib.utils import sigmoid


class SemanticLearner():
    
  def __init__(self, vUnits, lr = 0.00001):
    hUnits = 300
    self.W = np.random.normal(0, 0.01, (hUnits, vUnits))
    self.c = np.zeros(hUnits)        
    self.b = np.zeros(vUnits)
    self.lr = lr

  def forward(self, state):
    h = self.sampleHidden(state)
    return self.visibleProbability(h)
  
  def visibleProbability(self, h):
    return sigmoid(self.b + np.dot(h, self.W))

  def hiddenProbability(self, v):
    return sigmoid(self.c + np.dot(v, self.W.T)) 

  def sampleHidden(self, v):
    hProb = self.hiddenProbability(v)
    return np.random.uniform(size=hProb.shape) < hProb

  def backward(self, x):
    h1 = self.sampleHidden(x)
    pV2 = self.visibleProbability(h1)
    pH2 = self.hiddenProbability(pV2)
    
    # Update hidden units
    self.b += self.lr * (x - pV2)
    self.c += self.lr * (h1 - pH2)
    
    a = np.outer(h1.T, x)
    d = np.outer(pH2.T, pV2)
    self.W += self.lr * (a - d)
    error = np.sum(np.sum((x - pV2)**2))
