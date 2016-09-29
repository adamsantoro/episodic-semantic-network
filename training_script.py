from __future__ import division, print_function
import os  
import numpy as np
import matplotlib.pyplot as plt
from main.agent import Agent
from main.environment import Environment
from main.lib.utils import Visualizer
import optparse

parser = optparse.OptionParser()
parser.add_option('-d', '--delay', type=int, action="store", dest='delay', default=100)
options, args = parser.parse_args()
delay = options.delay


def train(trialMax = 5000, delay = 4000, policy = 1):
  agent = Agent.StandardAgent()
  env = Environment.UnimodalEnvironment(delay = delay)
  env.reset()
  state = env.state
  rewardsFound = 0

  while rewardsFound != 120:
    print('Finding reward # ', rewardsFound + 1, end='\r')
    action = agent.act(state, env.trialTime)
    state, reward = env.step(action)
    agent.learn(state, reward, env.delay)
    
    if reward == 1:
      rewardsFound += 1
      print('\nFound reward: ', rewardsFound)
      print('Time to reward: ', env.trialTime)
      agent.learn(state, reward, env.delay)
      agent.policy = 1
      env.reset()
      state = env.state

for i in range(20):
  train(delay = delay)


