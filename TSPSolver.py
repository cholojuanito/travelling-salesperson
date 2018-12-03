#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


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

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while not foundTour and time.time()-start_time < time_allowance:

            costMatrix = self.createOriginalCostMatrix(
                self._scenario.getCities())

            # for i in range(ncities):
            #     minCostExitIdx = np.unravel_index(
            #         costMatrix[i].argmin(), costMatrix[i].shape)
            #     for j in range(ncities):

            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            i = 0
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
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        totalStates = 1  # We always start with an intial state
        numSolutions = 0
        numPruned = 0
        bssf = self.defaultRandomTour()['soln']
        start_time = time.time()

        initCostMatrix = self.createOriginalCostMatrix(
            self._scenario.getCities())
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
        results['max'] = totalStates
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
