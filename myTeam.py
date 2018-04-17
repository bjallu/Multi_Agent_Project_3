# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
from capture import AgentRules
import random, time, util
from game import Directions
from util import nearestPoint
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'LuckyLuke', second = 'LuckyLuke'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class LuckyLuke(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.LayoutHalfWayPoint_Width = gameState.data.layout.width/2
    self.LayoutHalfWayPoint_Height = gameState.data.layout.height/2
    CaptureAgent.registerInitialState(self, gameState)
    self.grid_to_work_with = [g for g in gameState.getWalls().asList(False) if g[1] > 1]
    self.middle = (self.LayoutHalfWayPoint_Width, self.LayoutHalfWayPoint_Height)
    self.middle = self.find_place_in_grid(gameState, self.middle)
    # try features for the upper and lower half so the agent not in the middle could settle on either one of those points
    self.lowerHalf = (self.LayoutHalfWayPoint_Width, gameState.data.layout.height/2-0.25*self.LayoutHalfWayPoint_Height)
    self.lowerHalf = self.find_place_in_grid(gameState, self.lowerHalf)
    self.upperHalf = (self.LayoutHalfWayPoint_Width, gameState.data.layout.height/2+0.25*self.LayoutHalfWayPoint_Height)
    self.upperHalf = self.find_place_in_grid(gameState, self.upperHalf)
    #scaredTimer #maybe if scared run into the nearest player if he is x close to reset or hide? i dont know which is better
    self.enemy_indexes = self.getOpponents(gameState)
    #dictionary function inside this pacman thingy
    map = self.convert_tuples_to_list(self.grid_to_work_with)
    self.emission_probabilties_for_each_location_for_each_agent = []
    for enemy in self.enemy_indexes:
      locations_odd_per_player = [enemy, map]
      self.emission_probabilties_for_each_location_for_each_agent.append(locations_odd_per_player)
    print(self.emission_probabilties_for_each_location_for_each_agent[0][1][0])
    '''
    Your initialization code goes here, if you need any.
    '''

  def find_place_in_grid(self, gameState, location):
    #we only have half of the board to work with so
    w = gameState.data.layout.width/2
    #change the tuple to a list
    location_to_work_with = list(location)
    if location not in self.grid_to_work_with:
      location_copy = location_to_work_with
      for i in range(1, 3):
        if self.red:
          location_copy[0] = location_copy[0] - i
        else:
          location_copy[0] = location_copy[0] + i
        if tuple(location_copy) in self.grid_to_work_with:
          return tuple(location_copy)
      for i in range(1, 4):
        location_copy[1] = location_copy[1] - i
        if tuple(location_copy) in self.grid_to_work_with:
          return tuple(location_copy)
        location_copy[1] = location_copy[1] + i
        if tuple(location_copy) in self.grid_to_work_with:
          return tuple(location_copy)
      #Next one above
      #Third one below
      #Then two to left etc etc
      return (0, 0) #got to figure this edge case out
    return location
  def chooseAction(self, gameState):
    '''
    You should change this in your own agent.
    '''
    #dist = gameState.getAgentDistances()
    #enemies = self.getOpponents(gameState)
    #for daltonBrother in enemies:
    #  int = gameState.getInitialAgentPosition(daltonBrother)
    #  print(dist[daltonBrother])

    #mby add some function if no one is attacking us
    # Check if the enemy has any pacman.


    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    for action in actions:
      successor =self.getSuccessor(gameState, action)
      enemies = self.getOpponents(gameState)
      #team = self.getTeam(gameState)
      #enemy_kill_count = [AgentRules.checkDeath(gameState, i) for i in enemies]
      #team_kill_c = [AgentRules.checkDeath(gameState, i) for i in team]
      #print(enemy_kill_count)
      #print(team_kill_c)

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    squad = [successor.getAgentState(i) for i in self.getTeam(successor)]
    teamMembersPositions = [i.getPosition() for i in squad]
    #get Team Mates state and position
    both_defending = True
    for player in squad:
      if player.isPacman:
        both_defending = False

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    #    invaders = [a for a in self.enemies if
    #            gameState.getAgentState(a).isPacman]
    #    numInvaders = len(invaders)

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    for enemy in self.enemy_indexes:
      self.getMostProbableNoisyDistance(myPos, enemy, gameState)

    #for action in actions:
    #successor =self.getSuccessor(gameState, action)
    #enemies_2 = self.getOpponents(successor)
    #team = self.getTeam(gameState) figure out how to get kill info then we rush to get two pellets then turn back
    #enemy_kill_count = [AgentRules.checkDeath(successor, i) for i in enemies_2]
    #team_kill_c = [AgentRules.checkDeath(gameState, i) for i in team]
    #print(enemy_kill_count)
      #print(team_kill_c)
    #AgentRules.checkDeath(state, agentIndex)
    #heh = [d.checkDeath() for d in enemies]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    #print(myPos)
    # Judge it on how close it to the middle
    features['distanceToMiddle'] = self.getMazeDistance(myPos, self.middle)
    # try features for the upper and lower half so the agent not in the middle could settle on either one of those points
    features['DistanceToUpperHalf'] = self.getMazeDistance(myPos, self.upperHalf)
    features['DistanceToLowerHalf'] = self.getMazeDistance(myPos, self.lowerHalf)
    # if they are both on defense then punish if they are close together
    if both_defending:
      features['distanceFromEachOther'] = self.getMazeDistance(teamMembersPositions[0], teamMembersPositions[1])
    else:
      features['distanceFromEachOther'] = 0

    #Todo populate this feature with hmm observation data
    #features['noisyInvaderDistance'] = hmm(heh)
    #Extra features from baseLine that account for closeness to the middle of the field
    #Which I think would help them to be able to stop our invaders sooner
    # Also possibilty to work with the scared timer of our agent
    # if otherAgentState.scaredTimer <= 0:
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -9999, 'onDefense': 100, 'invaderDistance': -1000, 'stop': -100, 'reverse': -2, 'distanceToMiddle': -200, 'distanceFromEachOther': 200, 'DistanceToUpperHalf': -50, 'DistanceToLowerHalf': -50}

  def convert_tuples_to_list(self, tuple):
    l = []
    for i in tuple:
      element = list(i)
      l.append(element)
    return l

  def getMostProbableNoisyDistance(self, mypos, enemy_we_are_checking, gameState):
    distance_to_agents = gameState.getAgentDistances()
    distance_to_enemy = distance_to_agents[enemy_we_are_checking]
    #Only check possible locations of the enemy in question
    for g in self.grid_to_work_with:
      trueDistance = util.manhattanDistance(mypos, g)
      emissionModel = gameState.getDistanceProb(trueDistance, distance_to_enemy)
      #todo add check for:
      #we know that if the actual distance is equal to 5 or less we always get it as a true reading
 #     print(emissionModel)
    #keep track of previous probabilities