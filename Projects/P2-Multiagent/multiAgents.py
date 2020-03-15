# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        n_inf = float("-inf")
        positions = list(newPos)
        pellets = currentGameState.getFood().asList()

        if action == "Stop":
            return n_inf
        # Check if non scared ghost and pacman occupy same space
        for ghost in newGhostStates:
            if ghost.getPosition() == tuple(positions) and ghost.scaredTimer is 0:
                return n_inf

        pellet_dists = []
        # If no collision, find closest pellet
        for pellet in pellets:
            pellet_dists.append(-1 * manhattanDistance(
                (pellet[0], pellet[1]),
                (positions[0], positions[1])
                ))
        # print("pellet distance: " + str(pellet_dists))
        return max(pellet_dists)
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, depth, agent):
            if agent >= gameState.getNumAgents():
                depth -= 1
                agent = 0

            if depth is 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif agent is 0:
                return max_value(gameState, depth, agent)
            else:
                return min_value(gameState, depth, agent)

        def max_value(gameState, depth, agent):
            actions = gameState.getLegalActions(agent)
            if actions == None:
                return self.evaluationFunction(gameState)

            max = [float("-inf"), 'Stop']

            for action in actions:
                temp = value(gameState.generateSuccessor(agent, action), depth, agent + 1)

                if type(temp) == list:
                    temp = temp[0]
                else:
                    temp = temp
                if temp > max[0]:
                    max = [temp, action]

            return max

        def min_value(gameState, depth, agent):
            actions = gameState.getLegalActions(agent)
            if actions == None:
                return self.evaluationFunction(gameState)

            min = [float("inf"), 'Stop']

            for action in actions:
                temp = value(gameState.generateSuccessor(agent, action), depth, agent + 1)

                if type(temp) == list:
                    temp = temp[0]
                else:
                    temp = temp
                if temp < min[0]:
                    min = [temp, action]

            return min

        # return action not value
        return value(gameState, self.depth, 0)[1]
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, depth, agent, alpha, beta):
            if agent >= gameState.getNumAgents():
                depth -= 1
                agent = 0

            if depth is 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif agent is 0:
                return max_value(gameState, depth, agent, alpha, beta)
            else:
                return min_value(gameState, depth, agent, alpha, beta)

        def max_value(gameState, depth, agent, alpha, beta):
            actions = gameState.getLegalActions(agent)
            if actions == None:
                return self.evaluationFunction(gameState)

            max = [float("-inf"), 'Stop']

            for action in actions:
                temp = value(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)

                if type(temp) == list:
                    temp = temp[0]
                else:
                    temp = temp
                if temp > max[0]:
                    max = [temp, action]
                if temp > beta:
                    return [temp, action]
                if temp > alpha:
                    alpha = temp
            return max

        def min_value(gameState, depth, agent, alpha, beta):
            actions = gameState.getLegalActions(agent)
            if actions == None:
                return self.evaluationFunction(gameState)

            min = [float("inf"), 'Stop']

            for action in actions:
                temp = value(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)

                if type(temp) == list:
                    temp = temp[0]
                else:
                    temp = temp
                if temp < min[0]:
                    min = [temp, action]
                if temp < alpha:
                    return [temp, action]
                if temp < beta:
                    beta = temp
            return min

        # return action not value
        return value(gameState, self.depth, 0, float("-inf"), float("inf"))[1]

        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        return self.max_value(gameState, depth)[1]

    def max_value(self, gameState, depth, agent = 0):
        actions = gameState.getLegalActions(agent)

        if not actions or gameState.isWin() or depth >= self.depth:
            return self.evaluationFunction(gameState), "Stop"

        successor_cost = float('-inf')
        successor_action = "Stop"

        for action in actions:
            successor = gameState.generateSuccessor(agent, action)

            cost = self.getMinValue(successor, depth, agent + 1)[0]

            if cost > successor_cost:
                successor_cost = cost
                successor_action = action

        return successor_cost, successor_action

    def getMinValue(self, gameState, depth, agent):
        actions = gameState.getLegalActions(agent)

        if not actions or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState), None

        successor_costs = []

        for action in actions:
            successor = gameState.generateSuccessor(agent, action)

            cost = 0

            if agent == gameState.getNumAgents() - 1:
                cost = self.max_value(successor, depth + 1)[0]
            else:
                cost = self.getMinValue(successor, depth, agent + 1)[0]

            successor_costs.append(cost)

        return sum(successor_costs) / float(len(successor_costs)), None
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    What I did was incentize scared ghosts, eating food and keeping distance from ghosts
    Ghost with a distance of 3 or smaller carry a higher incentive. If they're
    scare pacman will prioritizeeating them and if they arent pacman will avoid.
    Pacman will also place priority on eating remaining food.
    """
    "*** YOUR CODE HERE ***"

    ghost_states = currentGameState.getGhostStates()
    food_pos = currentGameState.getFood().asList()
    pac_pos = list(currentGameState.getPacmanPosition())

    food_dist = []
    ghost_data = []
    score = currentGameState.getScore()

    for ghost in ghost_states:
        if ghost.scaredTimer is not 0:
            ghost_data.append([0,1])
        else:
            ghost_pos = ghost.getPosition()
            distance = manhattanDistance(
                (ghost_pos[0], ghost_pos[1]),
                (pac_pos[0], pac_pos[1]))
            ghost_data.append([distance, 0])

    for food in food_pos:
        food_dist.append(-1 * manhattanDistance(
            (food[0], food[1]), (pac_pos[0], pac_pos[1])))

    f_value1 = 0
    f_value2 = 0
    g_value = 0

    temp_list = food_dist
    if temp_list:
        f_value1 = max(temp_list)
        temp_list.remove(f_value1)
        if temp_list:
            f_value2 = max(temp_list)

    for ghost in ghost_data:
        if ghost[0] == 3 and ghost[1] == 0:
            g_value -= 10
        elif ghost[0] == 2 and ghost[1] == 0:
            g_value -= 20
        elif ghost[0] <= 1 and ghost[1] == 0:
            g_value -= 40
        elif ghost[0] <= 1 and ghost[1] == 1:
            g_value += 40

    return f_value1 + 2 * f_value2 + g_value + 5 * score - 5 * len(food_dist)
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
