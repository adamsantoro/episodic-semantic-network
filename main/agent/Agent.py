from __future__ import division
import numpy as np
from components import EpisodicLearner
from components import SemanticLearner
from components import NavigationLearner
from ..lib.utils import activatePolicy



class StandardAgent():
  def __init__(self, episodicUnits = 980):
    self.learners = []
    self.learners.append(EpisodicLearner.EpisodicLearner(episodicUnits))
    self.learners.append(SemanticLearner.SemanticLearner(episodicUnits))
    self.navigation = NavigationLearner.NavigationLearner(episodicUnits)

    # Episodic, semantic, and overall
    self.goals = [np.zeros(episodicUnits), np.zeros(episodicUnits), 
        np.zeros(episodicUnits)]

    # Policy to switch between episodic and semantic network
    self.tau = 2000
    self.policyN = 1
      


  def act(self, state, trialTime):
    # Episodic goal
    self.ca1Now = self.probeCa1(state)
    self.learners[0].forward(state)
    self.goals[0] = self.learners[0].memoryGoal()
    
    # Semantic goal
    self.goals[1] = self.learners[1].forward(self.ca1Now)

    # Overall goal
    self.goals[2] = (self.policyN * self.goals[0] + (1 - self.policyN) 
        * self.goals[1])

    # Choose an action based on goal and predictions from nav network
    self.navigation.choose(self.goals[2], self.ca1Now, trialTime)
    
    # Decay policy
    self.decayPolicy()

    return self.navigation.action


  def learn(self, state, reward, delay):
    # We need to take a step in the environment
    # before we learn so that the navigation net has
    # access to the next state given the action taken
    self.ca1Next = self.probeCa1(state)
    if reward == 1:
      self.learners[0].temporalDifference(reward)
      self.learners[0].updateCriticW()
      self.learners[0].lr = 0.01
      for i in range(100):
        self.learners[0].forward(state)
        self.learners[0].backward()
      for i in range(200):
        randomState = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
        self.ca1Next = self.probeCa1(randomState)
        self.learners[1].backward(self.ca1Next)
    elif reward == 0:
      self.learners[0].lr = self.learners[0].TDdelta * 0.1
      self.learners[0].backward()
      self.learners[0].temporalDifference(reward)
      self.learners[0].updateCriticW()
      self.navigation.learn(self.ca1Now, self.ca1Next, self.navigation.action)

    # Decay policy
    for i in range(delay):
      self.decayPolicy()


  def probeCa1(self, state):
    return self.learners[0].probeCa1(state)


  def decayPolicy(self):
    self.policyN = activatePolicy(self.policyN, self.tau)
        
        
        