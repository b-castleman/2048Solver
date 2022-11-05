from BaseAI import BaseAI
import Grid
import math
import numpy as np
import time

class IntelligentAgent(BaseAI):
    def __init__(self):
        self.usePruning = True
        self.maxTime = .2 * .90 # buffer time of 10% allowed

    """ Get the next move that the AI should perform"""
    def getMove(self, grid):
        self.startTime = time.process_time()

        # Decide recursion depth based on where in the game we are
        endChoice = self.getPreferredAvailableMoves(grid)[0] # default move choice

        # Iterative depthening search to maximize usefulness in the given time range
        for maxRecursionDepth in range(1,8):
            self.maxRecursionDepth = maxRecursionDepth

            if time.process_time() - self.startTime > self.maxTime:
                break

            decision = self.maximize(grid,1,float("-inf"),float("+inf")) # <cur state, heuristic, recursion depth,alpha,beta>

            if decision != None:
                endChoice = decision
            else:
                break

            ## just run once
            #if grid.getMaxTile() < 512:
            #    break

        # 0 stands for ”Up”
        # 1 stands for ”Down”
        # 2 stands for ”Left”
        # 3 stands for ”Right”
        return endChoice


    """ Maximize the opportunity for MY (user's) move """
    def maximize(self,grid,curRecursionDepth,alpha,beta):
        # return tuple of <state to do, utility function>

        possibleActions = self.getPreferredAvailableMoves(grid)

        # stopper: if we are exceeding depth or time, just end the search
        if time.process_time() - self.startTime > self.maxTime:
            return None

        if curRecursionDepth > self.maxRecursionDepth:
            return self.heuristicFunction(grid)

        # if empty, we instantly lose
        if len(possibleActions) == 0:
            return float('-inf') #self.heuristicFunction(grid) # should never run for the first max call as game manager will end the game

        # define output
        curMaxUtility = float('-inf')
        actionToDo = possibleActions[0][0]

        for action in possibleActions:
            curAction = action[0]
            newGrid = action[1]
            nxtUtility = self.minimize(newGrid,curRecursionDepth+1,alpha,beta)

            if nxtUtility == None:
                return None

            if nxtUtility > curMaxUtility:
                curMaxUtility = nxtUtility
                actionToDo = curAction

            # Alpha Beta Pruning
            if self.usePruning:
                # Update alpha if needed
                if curMaxUtility >= beta:
                    break

                if curMaxUtility > alpha:
                    alpha = curMaxUtility

        if time.process_time() - self.startTime > self.maxTime and curRecursionDepth == 1: # we're about to return the decision
            return None

        if curRecursionDepth == 1:
            return actionToDo
        else:
            return curMaxUtility


    """ Maximize the opportunity for the opponent (computer's) placement """
    def minimize(self,grid,curRecursionDepth,alpha,beta):
        possibleActions = grid.getAvailableCells()

        # if empty, computer instantly wants this (win condition for computer)
        if len(possibleActions) == 0:
            return float("-inf")

        if curRecursionDepth > self.maxRecursionDepth:
            return self.heuristicFunction(grid)

        netUtilityProbabilities = []
        # Iterate through all possible actions
        for numInserted in [2,4]:
            for action in possibleActions:
                # Stochastic Probability of Action Occurring
                probOfAction = 1/len(possibleActions)

                if(numInserted == 2):
                    probOfAction *= .9
                else: # numInserted == 4
                    probOfAction *= .1

                newGrid = grid.clone()
                newGrid.insertTile(action,numInserted)
                newGridResult = self.maximize(newGrid,curRecursionDepth+1,alpha,beta)
                if newGridResult == None:
                    return None

                netUtilityProbabilities.append(newGridResult * probOfAction)


        # Return expected utility probability
        return sum(netUtilityProbabilities)

    """ Get back valid user moves in their preferred order of ULDR (ideal for pruning)"""
    def getPreferredAvailableMoves(self,grid):
        moveList = grid.getAvailableMoves()

        # Just move the "down" action to the last index
        for i in range(len(moveList)-1):
            # If we're on the move integer of 1 (aka down), switch it with the next index
            if moveList[i][0] == 1:
                nextIdx = i+1
                tmp = moveList[nextIdx]
                moveList[nextIdx] = moveList[i]
                moveList[i] = tmp
                break

        return moveList # preferred order is now ULDR

    """ Get an estimated heuristic based on our current positioning"""
    def heuristicFunction(self,grid):
        """ First heuristic is based on the number of open squares
            Second heuristic is based on potential to merge the nearby numbers
            Third heuristic is based on the absolute differences of connecting blocks
            Fourth heuristic is based on the ordering of tiles (does it increase or is it equal at all times?)
            Fifth heuristic is the absolute value of tiles"""

        h1 = 0
        h2 = 0
        h3 = 0
        h4 = 0
        h5 = 0

        for i in range(grid.size):
            for j in range(grid.size):
                curVal = grid.getCellValue((i,j))
                neighborVals = [grid.getCellValue((i+1,j)),grid.getCellValue((i-1,j)),grid.getCellValue((i,j+1)),grid.getCellValue((i,j-1))]

                # h1: number of empty squares available
                if curVal == 0:
                    h1 += 1

                # h5: absolute value on a non-expoding exponential scale
                if curVal != 0:
                    h5 += 1.01**curVal #math.log2(curVal)

                # h4: strictly increasing check
                # Compare downwards (penalized if lower tiles are greater values)
                if neighborVals[0] != None and neighborVals[0] > curVal:
                    h4 += math.log2(neighborVals[0] - curVal)
                # Compare right (penalized if further right tiles are greater values)
                if neighborVals[2] != None and neighborVals[2] > curVal:
                    h4 += math.log2(neighborVals[2] - curVal)

                for neighbor in neighborVals:
                    # h2: potential merging
                    if curVal == neighbor and curVal != 0:
                        h2 += math.log2(curVal) #1

                    # h3: absolute differences
                    if neighbor == None: # skip nones
                        continue

                    dif = abs(curVal - neighbor)
                    if dif != 0:
                        h3 += math.log2(dif) # test out nonlog scale for larger disparty penalties (Dont think it worked)

        # h2 & h3 double count all connecting values
        h2 /= 2
        h3 /= 3

        # best: normal weights
        #Total
        #1024: 40
        #Total
        #2048: 32
        #Total
        #Above: 2
        # 39,33,1

        # 2h1 2h3:
        # 33,38,1
        # 41,38,3
        # 42,41

        # 2h1:
        # 44,34,2

        # 2h1 2h2 2h3
        # 51,21,1

        # 5h3
        # 53,19,0

        # 2h3
        # 54,26,0

        # 2h1 2h3 2h4
        # 27,23,0

        # 2h1 2h3 0h4
        # 38,22,0

        #2h1 2h3 .5h4
        # 50,34,1

        #2h1 2h5 2h3 .75h4
        # 54,23,1

        # 2h1,2h5,2h3
        # 38,32,0



        # User wants to maximize the heuristic, computer wants to minimize it
        # Kohler et al. (2019) recommends first and foremost to maximize open squares
        dividends = 2*h1 + h2 + h5 # (open squares) + (merge potential) + (abs value of tiles)
        penalties = 2*h3 + h4      # (abs difference of touching blocks) + (strictly increasing ordering)

        return dividends - penalties