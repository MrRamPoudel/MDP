from typing import Mapping
import numpy as np
import sys
import copy 


class MDP:
    def __init__(self,transitionProbability = 0.8, reward = -.04, policyFile =None, maxIteration = 20, gamma=.95):
        #set things required to solve the problem
        self.transitionProbability = transitionProbability
        self.allRewards = reward
        self.policyFile = policyFile

        #probability for going left and right
        self.otherProbability = float((1-transitionProbability)/2)
        self.rows, self.cols = (3,4)

        #South, West, North, East
        self.actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        self.utilityGrid = np.zeros((self.rows,self.cols), dtype=float)
        
        self.NextUtility = self.utilityGrid
        self.gamma = gamma
        self.wall = (1,1)
        self.terminalStates = {(0,3):1, (1,3):-1}
        self.maxIteration = maxIteration

    #Map the policy to 0,1,2,3 so it can be used with actions
    def __mapPolicy(self):
        if not isinstance(self.policyFile, type(None)):
            self.policyGrid = np.loadtxt(self.policyFile, delimiter=',',dtype=int)
            for index, value in np.ndenumerate(self.policyGrid):
                if value == 1:
                    self.policyGrid[index] = 2
                elif value == -1:
                     self.policyGrid[index] = 0
                elif value == 2:
                     self.policyGrid[index] = 3
                elif value == -2:
                     self.policyGrid[index] = 1
    '''Given a policy find the utility for the all the states.
    Returns the utility at lower left corner'''
    def policyEvaluation(self):
        self.__mapPolicy()
        for k in range(self.maxIteration):
            for index, value in np.ndenumerate(self.NextUtility):
                x, y = index[0], index[1]
                if (x,y) in self.terminalStates or (x,y) == self.wall:
                    continue
                self.NextUtility[x][y] = self.__calculateUtility(self.policyGrid[x][y], x, y)
            self.utilityGrid = copy.deepcopy(self.NextUtility)
        return self.utilityGrid[2][0]
    
    # get the utitlity with the reward
    def __getUtility(self,action, row, col):
        x, y = row, col
        newRow, newCol = self.actions[action]
        x += newRow
        y += newCol
        #terminalStates have 0 utility
        if (x,y) in self.terminalStates:
            return self.terminalStates[x,y]
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols or (x,y) == self.wall:
            return self.allRewards + self.gamma * self.utilityGrid[row][col]
        return self.allRewards + self.gamma * self.utilityGrid[x][y]

    def __calculateUtility(self, action, row, col):
        u = 0.0
        u += .8 * self.__getUtility(action, row, col)
        u += .1 * self.__getUtility((action -1)% 4, row, col)
        u += .1 * self.__getUtility((action + 1) % 4, row, col)
        return u


#main program
#call policy evaluation
reward = float(sys.argv[1])
filename = sys.argv[2]
program = MDP(reward=reward, policyFile=filename)

ans = program.policyEvaluation()

print(ans)