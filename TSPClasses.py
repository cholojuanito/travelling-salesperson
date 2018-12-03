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

class SearchState:
    '''
    Represents a state at any given moment while searching through
    all the possible solutions for the Travelling Saleperson problem

    Properties:
    id - UUID
    route - List of cities visited in order
    visitedCities - Set of all visited cities
    costMatrix - 2-D array with :
                        1) Rows representing the cost to leave a city
                        2) Columns representing the cost to enter a city
    bound - the "lowest" possible cost of the route up to this state

    '''
    def __init__(self, parentCostMatrix, exitCity = None, enterCity = None, parentRoute = list(), parentBound = 0):
        # Give the state an id so it can be found in the heap
        self._id = uuid.uuid4().hex
        self.route = parentRoute
        self.visitedCities = set()
        self._initializeVisitedCities()
        self.costMatrix = parentCostMatrix
        self.matrixLen = len(self.costMatrix[0])
        # Initialize the bound based on the previous matrix's bound
        if (enterCity is None):
            self.bound = parentBound
        else:
            self.bound = parentBound + parentCostMatrix[exitCity._index][enterCity._index]


    '''
    Initializes the set with all the city names that have
    been visited
    '''
    def _initializeVisitedCities(self):
        for i in range(len(self.route)):
            self.visitedCities.add(self.route[i]._name)

    '''
    Adds a city to the route.
    If it has been visited already then don't add
    it and return False so we skip this option
    '''
    def _addCityToRoute(self, newCity):
        if (newCity._name in self.visitedCities):
            return False
        else:
            self.route.append(newCity)
            self.visitedCities.add(newCity._name)
            return True

    '''
    Fills the row of the city we exit and the
    column of the city we enter with infinity
    This should prevent them from being visited again
    '''
    def _removeCitiesFromCostMatrix(self, enterCityIdx, exitCityIdx):
        for i in range(self.matrixLen):
            self.costMatrix[exitCityIdx][i] = math.inf
            self.costMatrix[i][enterCityIdx] = math.inf

        self.costMatrix[exitCityIdx, enterCityIdx] = math.inf	
        self.costMatrix[enterCityIdx, exitCityIdx] = math.inf

    '''
    Finds the minimum value in every row and subtracts
    it from each entry. 
    We then add that minimum value to the lower bound

    This represents the cost of leaving any city
    '''
    def rowReduce(self):
        for rowIdx in range(self.matrixLen):
            reduceAmount = self.costMatrix[rowIdx].min()
            if(reduceAmount < math.inf and reduceAmount > 0):
                self.bound = self.bound + reduceAmount
                for colIdx in range(self.matrixLen):
                    self.costMatrix[rowIdx][colIdx] = self.costMatrix[rowIdx][colIdx] - reduceAmount
    
    '''
    Finds the minimum value in every column and subtracts
    it from each entry. 
    We then add that minimum value to the lower bound

    This represents the cost of entering any city
    '''
    def colReduce(self):
        for colIdx in range(self.matrixLen):
            column = self.costMatrix[:, colIdx]
            reduceAmount = column.min()
            if(reduceAmount < math.inf and reduceAmount > 0):
                self.bound = self.bound + reduceAmount
                for rowIdx in range(self.matrixLen):
                    self.costMatrix[rowIdx][colIdx] = self.costMatrix[rowIdx][colIdx] - reduceAmount

    '''
    Overriden methods so that this class works with
    the heap in the correct way
    '''
    def __eq__(self, other):
        return self.bound == other.bound
    
    def __lt__(self, other):
        return self.bound < other.bound

    def __ne__(self, other):
        return not self.bound == other.bound