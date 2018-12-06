#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import copy
import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < math.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy( self,time_allowance=60.0):
        if (len(self._scenario.getCities()) == 0):
            return self.invalidResult()
        results = {}
        foundTour = False
        startCity = 0
        currentCity = startCity
        allCities = copy.deepcopy(self._scenario.getCities())
        citiesLeft = allCities
        bssf = TSPSolution([allCities[startCity]])
        tryNewStartCity = False
        route = []
        start_time = time.time()

        while not foundTour and time.time() - start_time < time_allowance:
            #for case of previous start city not finding a tour
            if(tryNewStartCity == True):
                #if all cities have been tried as start city
                if (startCity + 1 >= len(allCities)):
                    break
                #increment startCity and reset tour info
                startCity = startCity + 1
                route = []
                currentCity = startCity
                citiesLeft = self._scenario.getCities()
            #add current city to route and remove from list of possible cities to go to
            route.append(allCities[currentCity])
            citiesLeft.remove(allCities[currentCity])
            #find closest/lowest cost city to travel to next
            if (len(citiesLeft) > 0):
                closestNeighbor = 0
                closestCost = math.inf
                for i in range(len(citiesLeft)-1):
                    cost = allCities[currentCity].costTo(citiesLeft[i])
                    if (cost < closestCost):
                        closestCost = cost
                        closestNeighbor = i
                #switch currentCity to the closest city found
                currentCity = closestNeighbor
            else:
                # route.append(startCity) #don't need this because TSPSolution does it in cost function
                #create a solution because we visited all cities (greedy = stop on first valid solution)
                bssf = TSPSolution(route)
                if bssf.cost < math.inf:
                    foundTour = True
                else:
                    #doesn't count as a solution if cost is infinite. Try a new start city.
                    tryNewStartCity = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = len(route)
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def invalidResult(self):
        results = {}
        results['cost'] =  math.inf
        results['time'] = math.inf
        results['count'] = 0
        results['soln'] = None
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

    def branchAndBound(self, time_allowance=60.0):
        '''
        Time: O(n^2 B^n) where "n" is the number of cities and "B" is
            the "branching factor". The branching factor is a number
            that is calculated for each state. It is dependent on many
            factors including: 
                1) Depth in the search tree
                2) The number of cities that can be reached from that state

            The n in the exponent comes from the number of possible
            states that can be produced at that particular state. 

            The n^2 comes from the amount of work done while creating
            each state. At each state, a cost matrix is made and then
            reduced to obtain the bound for the state. This involves
            iterating over an 'n by n' 2-D array, thus giving us n^2 operations.

        Space: O(n2 E B) – where n is the number of  cities, E is the number of edges between the cities at a particular state and B is the branch factor mentioned above.
            The n2 represents the 2-D cost matrix that each state contains.
            The E represents the number of possible outward edges at any given state.
            This encompasses the size and amount of the states that could possibly be created.

        The Branch and Bound algorithm for solving the travelling 
        salesperson problem.
        This algorithm uses SearchState objects to represent a node
        in the search tree. The search tree is implemented using a 
        a min-heap.

        ## The Algorithm ##
        The basic idea is that we start with an initial state and 
        create/expand into child states from there. If at any point
        a child state contains a better solution than the current 
        bssf (best solution so far) we then update the bssf and now
        compare each child state that is made or state from the heap
        with the bssf. The bssf is intialized to a random permutation.
         If a state's bound is greater than the bssf's cost
        then we "prune" it from the search tree. Since logically it would
        be a waste of time to keep searching that branch.

        ## The Heap ##
        The heap used is a min-heap. It is implemented using the heapq
        module. It sorts the heap based on the bound of each state.
        The state with the lowest bound will always be the first element
        in the heap
        '''
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        totalStates = 1  # We always start with an intial state
        numSolutions = 0
        numPruned = 0
        maxHeapSize = 1
        bssf = self.defaultRandomTour()['soln']
        start_time = time.time()

        initCostMatrix = self.createOriginalCostMatrix(cities)
        heap = []
        heapq.heapify(heap)

        # Initialize the heap with a state
        initState = SearchState(initCostMatrix.copy(), cities[0])
        initState._addCityToRoute(cities[0])
        initState.rowReduce()
        initState.colReduce()

        heapq.heappush(heap, initState)

        while(len(heap) > 0 and (time.time() - start_time) < time_allowance):
            parentState = heapq.heappop(heap)
            # Check if the next state is even worth following
            if(parentState.bound < bssf.cost):
                parentCostMatrix = parentState.costMatrix
                # Make child states based on the exit city
                exitCity = cities[parentState.route[-1]._index]
                row = parentCostMatrix[exitCity._index]
                for i in range(ncities):
                    if(row.min() < math.inf):
                        cost = row[i]
                        if(cost < math.inf):
                            enterCity = cities[i]
                            # Create a new state from exitCity to enterCity
                            newState = SearchState(parentCostMatrix.copy(
                            ), enterCity=enterCity, exitCity=exitCity, parentRoute=parentState.route.copy(), parentBound=parentState.bound)

                            # Increase the total number of states made
                            totalStates += 1

                            # Make sure we don't add the final city twice
                            if (newState._addCityToRoute(enterCity) == True):
                                newState._removeCitiesFromCostMatrix(
                                    enterCity._index, exitCity._index)
                                newState.rowReduce()
                                newState.colReduce()

                                if(newState.bound < bssf.cost):
                                    heapq.heappush(heap, newState)

                                    # Increase the max heap size if needed
                                    if (len(heap) > maxHeapSize):
                                        maxHeapSize = len(heap)

                                    if(len(newState.route) == ncities and self._scenario._edge_exists[newState.route[-1]._index][newState.route[0]._index]):
                                        bssf = TSPSolution(newState.route)
                                        numSolutions += 1
                                else:
                                    # Increase the number of pruned "branches"
                                    numPruned += 1
                            else:
                                # Increase the number of pruned "branches"
                                numPruned += 1
            else:
                # Increase the number of pruned "branches"
                numPruned += 1

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = numSolutions
        results['soln'] = bssf
        results['max'] = maxHeapSize
        results['total'] = totalStates
        results['pruned'] = numPruned
        return results

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        pass

    def createOriginalCostMatrix(self, cities):
        ncities = len(cities)
        costMatrix = np.ndarray((ncities, ncities))
        for i in range(ncities):
            for j in range(ncities):
                costMatrix[i][j] = cities[i].costTo(cities[j])
        return costMatrix
