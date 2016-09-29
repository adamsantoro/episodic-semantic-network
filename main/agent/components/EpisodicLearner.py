from __future__ import division
import numpy as np
from ...lib.utils import sigmoid


class EpisodicLearner():

  def __init__(self, units, lr = 0.01):
    self.ca3Units = units
    self.ca1Units = units

    self.ca1Fields = [np.random.uniform(-0.4, 1.4, units), 
                      np.random.uniform(-0.4, 1.4, units)]

    # Stored activity patterns
    self.ca3A = [np.zeros(units), np.zeros(units), np.zeros(units)] 
    self.ca1A = [np.zeros(units), np.zeros(units)]
    self.critic = [0, 0]
    self.TDdelta = 0

    # Weights
    self.xyca3W = np.random.normal(0, 0.1, (2, units))
    self.ca3ca3W = np.random.normal(0, 0.1, (units, units))
    self.ca3ca1W = np.random.normal(0, 0.1, (units, units))
    self.ca1CriticW = np.zeros((units, 1))

    self.momentum = 0
    self.lr = lr


  def forward(self, state):
    # CA3        
    self.ca3A[0] = sigmoid(np.dot(state, self.xyca3W))
    self.ca3A[1] = sigmoid(np.dot(self.ca3A[0], self.ca3ca3W) - 1)
    self.ca3A[2] = sigmoid(np.dot(self.ca3A[1], self.ca3ca3W) - 1)
        
    # CA1 holds 2 states. The first is cued from the CA3. The second is
    # cued from space
    self.ca1A[0] = sigmoid(np.dot(self.ca3A[2], self.ca3ca1W))
    self.ca1A[1] = self.probeCa1(state)

    self.critic[0] = self.critic[1]
    self.critic[1] = self.activateCritic()


  def backward(self):
    # CA3 auto-encoding
    ca3error = self.ca3A[2] - self.ca3A[0]
    
    # Create derivative for each layer
    dEdca3A3 = self.ca3A[2] * (1 - self.ca3A[2])
    dEdca3A2 = self.ca3A[1] * (1 - self.ca3A[1])
        
    # Calculate node deltas 
    delta3 = dEdca3A3 * ca3error
    delta2 = np.sum(self.ca3ca3W * delta3, 1) * dEdca3A2
    
    # Update weights
    dW0 = -self.lr  * delta2 * self.ca3A[0].reshape(self.ca3Units, 1)
    dW1 = -self.lr  * delta3 * self.ca3A[1].reshape(self.ca3Units, 1)
    dWsum = (dW0 + dW1)/2

    self.ca3ca3W = self.ca3ca3W + dWsum + self.momentum
    self.momentum = 0.5 * dWsum

    # CA1
    ca1error = self.ca1A[0] - self.ca1A[1]
    dEdca1A = self.ca1A[0] * (1 - self.ca1A[0])
    delta = dEdca1A * ca1error

    # Update weights
    self.ca3ca1W += - self.lr * delta * self.ca3A[2].reshape(self.ca3Units, 1)


  def probeCa1(self, state):
    return (np.exp(-((state[0] - self.ca1Fields[0])**2 
        + (state[1] - self.ca1Fields[1])**2) / (2 * (0.16**2))))


  def memoryGoal(self):
    return self.ca1A[0]


  def activateCritic(self):
    return np.dot(self.ca1A[1], self.ca1CriticW)


  def temporalDifference(self, reward):
    if reward != 1:
      self.TDdelta = 0.95 * self.critic[1] - self.critic[0]
    elif reward == 1:
      self.TDdelta = reward - self.critic[0]


  def updateCriticW(self):
    return self.ca1CriticW + 0.01 * self.TDdelta * self.ca1A[1].reshape(self.ca1A[1].size, 1)

