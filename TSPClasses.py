#!/usr/bin/python3


import math
import numpy as np
import random
import time
import uuid



class TSPSolution:
    def __init__( self, listOfCities):
        self.route = listOfCities
        self.cost = self._costOfRoute()
        #print( [c._index for c in listOfCities] )

    def _costOfRoute( self ):
        cost = 0
        #print('cost = ',cost)
        last = self.route[0]
        for city in self.route[1:]:
            #print('cost increasing by {} for leg {} to {}'.format(last.costTo(city),last._name,city._name))
            cost += last.costTo(city)
            last = city
        #print('cost increasing by {} for leg {} to {}'.format(self.route[-1].costTo(self.route[0]),self.route[-1]._name,self.route[0]._name))
        cost += self.route[-1].costTo( self.route[0] )
        #print('cost = ',cost)
        return cost

    def enumerateEdges( self ):
        elist = []
        c1 = self.route[0]
        for c2 in self.route[1:]:
            dist = c1.costTo( c2 )
            if dist == np.inf:
                return None
            elist.append( (c1, c2, int(math.ceil(dist))) )
            c1 = c2
        dist = self.route[-1].costTo( self.route[0] )
        if dist == np.inf:
            return None
        elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
        return elist


def nameForInt( num ):
    if num == 0:
        return ''
    elif num <= 26:
        return chr( ord('A')+num-1 )
    else:
        return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)








class Scenario:

    HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges

    def __init__( self, city_locations, difficulty, rand_seed ):
        self._difficulty = difficulty

        if difficulty == "Normal" or difficulty == "Hard":
            self._cities = [City( pt.x(), pt.y(), \
                                  random.uniform(0.0,1.0) \
                                ) for pt in city_locations]
        elif difficulty == "Hard (Deterministic)":
            random.seed( rand_seed )
            self._cities = [City( pt.x(), pt.y(), \
                                  random.uniform(0.0,1.0) \
                                ) for pt in city_locations]
        else:
            self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


        num = 0
        for city in self._cities:
            #if difficulty == "Hard":
            city.setScenario(self)
            city.setIndexAndName( num, nameForInt( num+1 ) )
            num += 1

        # Assume all edges exists except self-edges
        ncities = len(self._cities)
        self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

        #print( self._edge_exists )
        if difficulty == "Hard":
            self.thinEdges()
        elif difficulty == "Hard (Deterministic)":
            self.thinEdges(deterministic=True)

    def getCities( self ):
        return self._cities


    def randperm( self, n ):				#isn't there a numpy function that does this and even gets called in Solver?
        perm = np.arange(n)
        for i in range(n):
            randind = random.randint(i,n-1)
            save = perm[i]
            perm[i] = perm[randind]
            perm[randind] = save
        return perm

    def thinEdges( self, deterministic=False ):
        ncities = len(self._cities)
        edge_count = ncities*(ncities-1) # can't have self-edge
        num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

        #edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0
        can_delete	= self._edge_exists.copy()

        # Set aside a route to ensure at least one tour exists
        route_keep = np.random.permutation( ncities )
        if deterministic:
            route_keep = self.randperm( ncities )
        for i in range(ncities):
            can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

        # Now remove edges until 
        while num_to_remove > 0:
            if deterministic:
                src = random.randint(0,ncities-1)
                dst = random.randint(0,ncities-1)
            else:
                src = np.random.randint(ncities)
                dst = np.random.randint(ncities)
            if self._edge_exists[src,dst] and can_delete[src,dst]:
                self._edge_exists[src,dst] = False
                num_to_remove -= 1

        #print( self._edge_exists )




class City:
    def __init__( self, x, y, elevation=0.0 ):
        self._x = x
        self._y = y
        self._elevation = elevation
        self._scenario	= None
        self._index = -1
        self._name	= None

    def setIndexAndName( self, index, name ):
        self._index = index
        self._name = name

    def setScenario( self, scenario ):
        self._scenario = scenario

    ''' <summary>
        How much does it cost to get from this city to the destination?
        Note that this is an asymmetric cost function.
         
        In advanced mode, it returns infinity when there is no connection.
        </summary> '''
    MAP_SCALE = 1000.0
    def costTo( self, other_city ):

        assert( type(other_city) == City )

        # In hard mode, remove edges; this slows down the calculation...
        # Use this in all difficulties, it ensures INF for self-edge
        if not self._scenario._edge_exists[self._index, other_city._index]:
            #print( 'Edge ({},{}) doesn\'t exist'.format(self._index,other_city._index) )
            return np.inf

        # Euclidean Distance
        cost = math.sqrt( (other_city._x - self._x)**2 +
                          (other_city._y - self._y)**2 )

        # For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
        if not self._scenario._difficulty == 'Easy':
            cost += (other_city._elevation - self._elevation)
            if cost < 0.0:
                cost = 0.0
        #cost *= SCALE_FACTOR


        return int(math.ceil(cost * self.MAP_SCALE))

class PriorityQueueHeap:
    '''
    A min-heap implementation of a priority queue to be used in the Travelling Salesperson Algorithm
    '''

    def __init__(self):
        self.queue = list()
        self.queueIndexMap = dict()
        self.maxSize = 0

    def insert(self, state):
        '''
        Time Complexity: O(log|V|) because we call siftUp

        Adds a state and sorts by calling siftUp

        The key for sorting the heap is based on both how deep into
        the search tree we are and how small the lower boud for the
        state is. I figured the average length between each city in the 
        current state would be the best way to represent this.
        '''
        self.queue.append([state, (state.bound / len(state.route))])
        self.queueIndexMap[state._id] = len(self.queue) - 1
        self.siftUp(state)
        if (self.maxSize < len(self.queue)):
            self.maxSize = len(self.queue)

    def deleteMin(self):
        '''
        Time Complexity: O(log|V|) because we call siftDown

        Swaps the first state in the heap with the last one.
        Then calls siftDown
        '''
        # Delete the first one and return it
        lastIndex = len(self.queue) - 1
        bestState = self.queue[0][0]
        lastState = self.queue[lastIndex][0]

        # Change positions with very last element
        self.updateMap(bestState._id, lastState._id)
        self.updateQueue(0, lastIndex)

        # Delete the bestState - now in the last element
        del self.queue[lastIndex]

        # Send the state that was last in to be siftted down
        if(len(self.queue) > 0):
            self.siftDown(self.queue[0][0])

        return bestState

    def siftUp(self, nodeToSift):
        '''
        Time Complexity: O(log|V|) because the farthest we will go up the 'tree' is
        to the top. Assuming we start at the bottom then we will be making log|V|
        iterations until we reach the top of the 'tree'

        Space Complexity: N/A because we are simply using the queue and map that
        have already been created

        Iteratively compares a node to its 'parent' in the 'tree'.
        If the parent node's key is less than the current node's then they are swapped
        '''
        if(len(self.queue) < 2):
            return

        while(True):
            indexNodeToSift = self.queueIndexMap[nodeToSift._id]
            nodeToSiftVal = self.queue[indexNodeToSift][1]
            indexParentNode = indexNodeToSift // 2 if (
                indexNodeToSift % 2 == 1) else int((indexNodeToSift / 2) - 1)
            if(indexParentNode < 0):
                indexParentNode = 0

            parentNodeVal = self.queue[indexParentNode][1]
            if (parentNodeVal > nodeToSiftVal):
                parentNode = self.queue[indexParentNode][0]
                self.updateMap(nodeToSift._id, parentNode._id)
                self.updateQueue(indexNodeToSift, indexParentNode)
            
            #TODO Maybe do an else if for a tie breaker?
            else:
                break

    def siftDown(self, nodeToSift):
        '''
        Time Complexity: O(log|V|) because the farthest we will go down the 'tree' is
        to the bottom. We always start at the top of the 'tree' we make at most log|V|
        iterations until we reach the bottom of the 'tree'

        Space Complexity: N/A because we are simply using the queue and map that
        have already been created, assuming we put aside all the local variables I use
        in this function

        Iteratively compares a node's key with the keys of its 'children' in the 'tree'.
        It picks the smaller of the two if there are two 'children'.

        If the child's key is smaller than node's key then they are swapped
        '''
        if(len(self.queue) < 2):
            return

        while(True):
            indexNodeToSift = self.queueIndexMap[nodeToSift._id]
            leftChildIndex = (2 * indexNodeToSift) + 1
            rightChildIndex = (2 * indexNodeToSift) + 2
            lastIndex = len(self.queue) - 1
            # Check for overflow
            if (leftChildIndex > lastIndex):
                leftChildIndex = lastIndex
            if (rightChildIndex > lastIndex):
                rightChildIndex = lastIndex

            leftChildNode = self.queue[leftChildIndex][0]
            rightChildNode = self.queue[rightChildIndex][0]
            leftChildNodeVal = self.queue[leftChildIndex][1]
            rightChildNodeVal = self.queue[rightChildIndex][1]
            nodeToSiftVal = self.queue[indexNodeToSift][1]

            # Check if left child is viable
            if (leftChildNodeVal < nodeToSiftVal and leftChildNodeVal <= rightChildNodeVal):
                self.updateMap(nodeToSift._id, leftChildNode._id)
                self.updateQueue(indexNodeToSift, leftChildIndex)

                # else if the right child is viable
            elif (rightChildNodeVal < nodeToSiftVal and rightChildNodeVal <= leftChildNodeVal):
                self.updateMap(nodeToSift._id, rightChildNode._id)
                self.updateQueue(indexNodeToSift, rightChildIndex)
                # else should be done
            else:
                break

    def updateMap(self, nodeToSiftId, otherNodeId):
        '''
        Time Complexity: O(1)

        Space Complexity: N/A

        Simply swaps the values
        '''
        self.queueIndexMap[nodeToSiftId], self.queueIndexMap[otherNodeId] = self.queueIndexMap[otherNodeId], self.queueIndexMap[nodeToSiftId]

    def updateQueue(self, indexNodeToSift, indexOtherNode):
        '''
        Time Complexity: O(1)

        Space Complexity: N/A

        Simply swaps the values
        '''
        self.queue[indexNodeToSift], self.queue[indexOtherNode] = self.queue[indexOtherNode], self.queue[indexNodeToSift]


'''
Comments go here
'''
class SearchState:
    def __init__(self, parentCostMatrix, enterCity = None, exitCity = None, parentRoute = list(), parentBound = 0):
        # Give the state an id so it can be found in the heap
        self._id = uuid.uuid4().hex
        self.route = parentRoute
        self.visitedCities = set()
        self._initializeVisitedCities()
        self.costMatrix = parentCostMatrix
        self.matrixLen = len(self.costMatrix[0])
        # Initialize the bound based on the previous matrix's bound
        self.bound = parentBound
    
    def _initializeVisitedCities(self):
        for i in range(len(self.route)):
            self.visitedCities.add(self.route[i]._name)

    def _addCityToRoute(self, newCity):
        if (newCity._name in self.visitedCities):
            return False
        else:
            self.route.append(newCity)
            self.visitedCities.add(newCity._name)
            return True

    def _removeCitiesFromCostMatrix(self, enterCityIdx, exitCityIdx):
        for i in range(self.matrixLen):
            self.costMatrix[exitCityIdx][i] = math.inf
            self.costMatrix[i][enterCityIdx] = math.inf

        self.costMatrix[exitCityIdx, enterCityIdx] = math.inf	
        self.costMatrix[enterCityIdx, exitCityIdx] = math.inf

    def rowReduce(self):
        for rowIdx in range(self.matrixLen):
            reduceAmount = self.costMatrix[rowIdx].min()
            if(reduceAmount < math.inf and reduceAmount > 0):
                self.bound = self.bound + reduceAmount
                for colIdx in range(self.matrixLen):
                    self.costMatrix[rowIdx][colIdx] = self.costMatrix[rowIdx][colIdx] - reduceAmount
    
    def colReduce(self):
        for colIdx in range(self.matrixLen):
            column = self.costMatrix[:, colIdx]
            reduceAmount = column.min()
            if(reduceAmount < math.inf and reduceAmount > 0):
                self.bound = self.bound + reduceAmount
                for rowIdx in range(self.matrixLen):
                    self.costMatrix[rowIdx][colIdx] = self.costMatrix[rowIdx][colIdx] - reduceAmount
