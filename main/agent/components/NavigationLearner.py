from __future__ import division
import numpy as np
from ...lib.utils import sigmoid
from ...lib.utils import convertToProbability


class NavigationLearner():

  def __init__(self, units, lr = 0.05):
    self.W0 = np.random.normal(0, 0.1, (units + 8, 100))
    self.W1 = np.random.normal(0, 0.1, (100, units))
    self.actor = np.zeros(8)
    self.action = 0
    self.ca1 = np.zeros(units)
    self.lr = lr


  def forward(self):
    self.actor.fill(0)
    self.actor[self.action] = 1
    self.h = sigmoid(np.dot(np.append(self.ca1, self.actor), self.W0) + 1)
    return sigmoid(np.dot(self.h, self.W1) + 1)

  # Choosing always comes before learning, so self.ca1 state should always
  # be correct when calling forward during learning
  def choose(self, memoryGoal, ca1, trialTime):
    self.ca1 = ca1[:]
    hamming = np.zeros(8)
    for i in range(8):
      self.action = i
      output = self.forward()
      hamming[i] = np.sum(np.absolute(memoryGoal - output))
    prob = convertToProbability(1 - hamming, np.clip(trialTime/2000, 0, 0.99), 1)
    self.action = np.random.choice(np.arange(8), p=prob)

  def learn(self, current, target, action):
    self.ca1 = current
    self.action = action
    output = self.forward()
    error = output - target

    # Create derivative for each layer, without bias units
    dEdA2 = self.h * (1 - self.h)
    dEdA3 = output * (1 - output)

    # Calculate node deltas 
    delta3 = dEdA3 * error
    delta2 = np.sum(self.W1 * delta3, 1) * dEdA2

    # Update weights
    self.W0 = self.W0 - self.lr * delta2 * np.append(self.ca1, 
        self.actor).reshape(np.append(self.ca1, self.actor).size, 1)
    self.W1 = self.W1 - self.lr * delta3 * self.h.reshape(self.h.size, 1)

    self.output = output
