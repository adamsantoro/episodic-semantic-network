from __future__ import division
import numpy as np
import math


class UnimodalEnvironment():
  def __init__(self, mu = [0.7, 0.7], var = 0.1, delay = 4000, trialMax = 5000, 
    closeness = 0.05):
      
    self.state = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
    
    # Need a copy here to prevent prevState from changing when state changes
    self.prevState = self.state[:]
    self.mu = mu
    self.var = var
    self.delay = delay
    self.rewardLocation = [
        np.clip(np.random.normal(self.mu[0], self.var, 1), 0 , 1), 
        np.clip(np.random.normal(self.mu[1], self.var, 1), 0 , 1)]

    self.trialTime = 0
    self.rewardTime = 0
    
    self.trialMax = trialMax
    self.closeness = closeness
    
    # Actor directions. Order: E, NE, N, NW, W, SW, S, SE
    angles = np.arange(0, 2*np.pi + np.pi / 4, np.pi / 4)
    self.directions = [0.04 * np.cos(angles), 0.04 * np.sin(angles)]
    self.prevDirection = [0, 0]


  def step(self, action):
    self.trialTime += 1
    self.updateAgentLocation(action)
    self.bounceOffWall()

    if math.hypot(self.state[0] - self.rewardLocation[0], self.state[1] 
        - self.rewardLocation[1]) < self.closeness or self.trialTime > self.trialMax:
            reward = 1
            self.state[0] = self.rewardLocation[0][0]
            self.state[1] = self.rewardLocation[1][0]
    else:
        reward = 0

    self.updateRewardLocation()

    # Need a copy here to prevent prevState from changing when state changes
    self.prevState = self.state[:]

    return self.state, reward


  def reset(self):
    for i in range(self.delay):
      self.updateRewardLocation()            
    self.state = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
    self.prevState = self.state[:]
    self.trialTime = 0


  def updateRewardLocation(self):
    if self.rewardTime < 4000:
      self.rewardTime += 1
      self.rewardLocation[0] = np.clip(
        self.rewardLocation[0] + np.random.normal(0, 0.003), 0, 1)
      self.rewardLocation[1] = np.clip(
        self.rewardLocation[1] + np.random.normal(0, 0.003), 0, 1)
    else:
      self.rewardTime = 0
      self.rewardLocation = [
        np.clip(np.random.normal(self.mu[0], self.var, 1), 0 , 1), 
        np.clip(np.random.normal(self.mu[1], self.var, 1), 0 , 1)]


  def updateAgentLocation(self, action):
    self.state[0] = np.clip(self.state[0] + (3 * self.prevDirection[0] 
      + self.directions[0][action])/4, 0, 1)
    self.state[1] = np.clip(self.state[1] + (3 * self.prevDirection[1] 
       + self.directions[1][action])/4, 0, 1)
      

  def bounceOffWall(self):
    if self.state[0] > 0.99:
      self.prevDirection[0] = -0.03
    elif self.state[0] < 0.01:
      self.prevDirection[0] = 0.03
    else:
      self.prevDirection[0] = (self.state[0] - self.prevState[0])

    if self.state[1] > 0.99:
      self.prevDirection[1] = -0.03
    elif self.state[1] < 0.01:
      self.prevDirection[1] = 0.03
    else:
      self.prevDirection[1] = (self.state[1] - self.prevState[1])

        