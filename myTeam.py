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
from game import Directions, Grid
from util import nearestPoint
from captureGraphicsDisplay import PacmanGraphics
#if isinstance(self.display, PacmanGraphics):
import numpy as np

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

    #find if there is a pacman and this list has changed def getFoodYouAreDefending(self, gameState):

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''

    self.LayoutHalfWayPoint_Width = gameState.data.layout.width/2
    self.LayoutHalfWayPoint_Height = gameState.data.layout.height/2
    CaptureAgent.registerInitialState(self, gameState)
    grid_fiesta = []

    grid = np.zeros((gameState.data.layout.width, gameState.data.layout.height))
    for x in range(gameState.data.layout.width):
      for y in range(gameState.data.layout.height):
        if not gameState.hasWall(x, y):
          grid[x][y] = 1
          grid_fiesta.append((x, y))

    self.grid_to_work_with = grid_fiesta
    #

    self.middle = (self.LayoutHalfWayPoint_Width, self.LayoutHalfWayPoint_Height)
    self.middle = self.find_place_in_grid(gameState, self.middle)
    # try features for the upper and lower half so the agent not in the middle could settle on either one of those points
    self.lowerHalf = (self.LayoutHalfWayPoint_Width, gameState.data.layout.height/2-0.25*self.LayoutHalfWayPoint_Height)
    self.lowerHalf = self.find_place_in_grid(gameState, self.lowerHalf)
    self.upperHalf = (self.LayoutHalfWayPoint_Width, gameState.data.layout.height/2+0.25*self.LayoutHalfWayPoint_Height)
    self.upperHalf = self.find_place_in_grid(gameState, self.upperHalf)
    #scaredTimer #maybe if scared run into the nearest player if he is x close to reset or hide? i dont know which is better
    self.enemy_indexes = self.getOpponents(gameState)
    self.team_indexes = self.getTeam(gameState)
    min_value = min(self.enemy_indexes)
    if min == 0:
      self.enemy_to_update_possible_locations = self.index - 1
    else:
      self.enemy_to_update_possible_locations = self.index + 1
    self.map = self.convert_tuples_to_list(self.grid_to_work_with)

    self.invaders_must_die = []

    ## TODO ###
    #Make this code somewhat more smarter and more intuitive but it gets the job done for now
    self.emission_probabilties_for_each_location_for_each_agent = []
    for enemy in self.enemy_indexes:
      all_locations = self.map
      all_locations_plus_odds = []
      for location in all_locations:
        element = [location, 0]
        all_locations_plus_odds.append(element)
      self.emission_probabilties_for_each_location_for_each_agent.append(all_locations_plus_odds)
      # Because we know the inital position we can put the probabilty of the enemy being in that position to 100% i.e one
      initPos = list(gameState.getInitialAgentPosition(enemy))
      initPos = [initPos, 0]
      agent_list_index = self.getEnemyListIndex(enemy)
      starting_location_index = self.emission_probabilties_for_each_location_for_each_agent[agent_list_index].index(initPos)
      self.emission_probabilties_for_each_location_for_each_agent[agent_list_index][starting_location_index][1] = 1.0

    #print(self.emission_probabilties_for_each_location_for_each_agent[0][1][0])
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
    #mby add some function if no one is attacking us
    # Check if the enemy has any pacman.

    #TODOOO ADD DECAYING FACTOR / BIAS I:E FAVOUR POSITONS THAT ARE FURTHER AWAY FROM THEIR INITAL POSITON
    # AND CLOSER TO OUR HALFWAY POINT
    actions = gameState.getLegalActions(self.index)
    pos = gameState.getAgentState(self.index).getPosition()

    self.update_enemy_possible_locations_depending_on_round(self.enemy_to_update_possible_locations, gameState)
    #for enemy in self.enemy_indexes:
    #  index = self.getEnemyListIndex(enemy)
    #  list_to_print = [i for i in self.emission_probabilties_for_each_location_for_each_agent[index] if i[1] != 0]
    #  ls = [tuple(i[0]) for i in list_to_print]
    #  if index == 0:
    #    self.debugDraw(ls, [1, 0, 0], True)
    #  else:
    #    self.debugDraw(ls, [0, 1, 0], True)

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
      successor = self.getSuccessor(gameState, action)
      enemies = self.getOpponents(gameState)
      #team = self.getTeam(gameState)
      #enemy_kill_count = [AgentRules.checkDeath(gameState, i) for i in enemies]
      #team_kill_c = [AgentRules.checkDeath(gameState, i) for i in team]
      #print(enemy_kill_count)
      #print(team_kill_c)

    #fuond some debuggining thing
    ##  self.displayDistributionsOverPositions()
    #debug draw
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """


    #FIX THIS TO WORK WITH ALL THE GRID

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
    #print(myState.observationHistory)
    myPos = myState.getPosition()
    squad = [successor.getAgentState(i) for i in self.getTeam(successor)]
    teamMembersPositions = [i.getPosition() for i in squad]
    otherDudePos = 0
    if len(squad) > 1:
      #print(squad)
      # can use id check here maybe takes care of this wierd problem
      otherDude = [i for i in squad if i != myState]
      # I still don't know wwhy I need this check but we need it
      if len(otherDude) > 0:
        otherDude = otherDude[0]
        otherDudePos = otherDude.getPosition()
    #get Team Mates state and position
    both_defending = True
    for player in squad:
      if player.isPacman:
        both_defending = False

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    old_invaderList = self.invaders_must_die
    new_invaderList = [a for a in enemies if a.isPacman]
    self.invaders_must_die = new_invaderList
    # we killed some one
    if len(new_invaderList) < len(old_invaderList):
      #get that agent and reset his probabilty pos to the inital position
      enemies_to_reset_locations_for = list(set(old_invaderList) - set(new_invaderList))
      enemy_indexes = self.enemy_indexes
      enemy_indexes_to_reset = []
      if len(enemies_to_reset_locations_for) > 0:
        for index in enemy_indexes:
          index_to_store = index
          state = gameState.getAgentState(index)
          for lad in enemies_to_reset_locations_for:
            if state == lad:
              enemy_indexes_to_reset.append(index_to_store)
      for enemy in enemy_indexes_to_reset:
        # to do find index of the enemy that just diededed
        initPos = gameState.getInitialAgentPosition(enemy)
        initPos = [initPos[0], initPos[1]]
        self.reset_agent_probabilties_when_we_know_the_true_position(enemy, initPos)
    #Now we update the noisy distances for those that arent within our sensor range

    #### NEED TO FIGURE OUT JIUST TO CALL THIS ONCE PER TURN
    # or
    # SMART BJARTUR we assign an enemy to update to each agent indepentently


    #  IF A PELLET dissappears we know in legalmoves from that location
    # that a pacman is there
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      #print(dists)
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

    '''

    list_of_enemies_to_check_noisy_distance_for = [i for i in self.enemy_indexes if successor.getAgentState(i).getPosition() == None]
    list_of_enemies_in_range = [i for i in self.enemy_indexes if i not in list_of_enemies_to_check_noisy_distance_for]

    for i in list_of_enemies_in_range:
      self.reset_agent_probabilties_when_we_know_the_true_position(i, list(successor.getAgentState(i).getPosition()))

    #reset_agent_probabilties_when_we_know_the_true_position

    #if the distance is small under 2 and then not next round he prolly died and we reset the probabilty and location

    # need to add a check first to see if we have the true position of some of our agents
    list_of_most_probable_locations = []
    #enemies_within_sensor_range_position = [e for e in enemies if e.getPosition() != None]

    #### NEED TO FIGURE OUT JIUST TO CALL THIS ONCE PER TURN
    # or
    # SMART BJARTUR we assign an enemy to update to each agent indepentently
    self.update_enemy_possible_locations_depending_on_round(self.enemy_to_update_possible_locations, gameState)

    for enemy in self.enemy_indexes:
      #Take both of our agents into account
      self.updateNoisyDistanceProbabilities(myPos, enemy, gameState)
      if otherDudePos != 0:
        self.updateNoisyDistanceProbabilities(otherDudePos, enemy, gameState)
      list_of_most_probable_locations.append(self.get_most_likely_distance_from_noisy_reading(enemy))

    #todo check this bjartur something fhishy somtiems it returns zero also add the bias for places closer to us that
    # will get higher rating

    #Now get the enemy distances
    final_enemy_distances = []
    for enemy in self.enemy_indexes:
      if successor.getAgentState(enemy).getPosition() != None:
        final_enemy_distances.append(successor.getAgentState(enemy).getPosition())
      else:
        index = self.getEnemyListIndex(enemy)
        final_enemy_distances.append(list_of_most_probable_locations[index])

    #self.debugDraw(final_enemy_distances,[1,0,0],False)

    for enemy in self.enemy_indexes:
      index = self.getEnemyListIndex(enemy)
      list_to_print = [i for i in self.emission_probabilties_for_each_location_for_each_agent[index] if i[1] != 0]
      ls = [tuple(i[0]) for i in list_to_print]
      #print(ls)
      if index == 0:
        self.debugDraw(ls, [1,0,0], True)
      else:
        self.debugDraw(ls, [0, 1, 0], True)
    #self.debugDraw(final_enemy_distances,[1,0,0],False)

    distance_to_each_enemy_most_probable_location = [self.getMazeDistance(myPos, i) for i in list_of_most_probable_locations]
    if len(distance_to_each_enemy_most_probable_location) != 0:
      features['noisyInvaderDistance'] = min(distance_to_each_enemy_most_probable_location)
    #if we dont get the true distance we turn to the most probable noisy distance using hmms n stuff
    # Also possibilty to work with the scared timer of our agent
    # if otherAgentState.scaredTimer <= 0:
    '''
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -9999, 'onDefense': 100, 'invaderDistance': -1000, 'stop': -100, 'reverse': -2, 'distanceToMiddle': -200, 'distanceFromEachOther': 200, 'DistanceToUpperHalf': -50, 'DistanceToLowerHalf': -50}

  def convert_tuples_to_list(self, tuple):
    l = []
    for i in tuple:
      element = list(i)
      l.append(element)
    return l

  def getEnemyListIndex(self, enemy_we_are_checking):
    max_element = max(self.enemy_indexes)
    if enemy_we_are_checking == max_element:
      index = 1
    else:
      index = 0
    return index

  #if our enemy is a pacman we always know it true distance? to check

  #noisyDistances = gameState.getAgentDistances()

  ## Todo we know after each action our enemies could only have moved a single step in some possible action
  def update_enemy_possible_locations_depending_on_round(self, enemy_to_update, gameState):
    # all legal actions for the enemy depending on possible locations
    list_index = self.getEnemyListIndex(enemy_to_update)
    #    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    all_locations_not_with_zero_probabilty = [i for i in self.emission_probabilties_for_each_location_for_each_agent[list_index] if i[1] != 0]
    all_possible_moves = []
    for i in all_locations_not_with_zero_probabilty:
      posible_moves_from_i = self.get_legal_actions_from_location(i[0])
      all_possible_moves.extend(posible_moves_from_i)
    #Give all possible moves equal probability and then multiply those with odds
    number_of_possible_locations_to_move_to = len(all_possible_moves)
    for move in all_possible_moves:
      #find index of possible moves
      counter = 0
      for i in self.emission_probabilties_for_each_location_for_each_agent[list_index]:
        if i[0] == move:
          i[1] = 1/number_of_possible_locations_to_move_to

  def get_legal_actions_from_location(self, location):
    location = list(location)
    list_of_possible_coordinates = []
    #always possible to stop in place
    list_of_possible_coordinates.append(location)
    #now check up, down, left, right,
    up = [location[0], location[1]+1]
    down = [location[0], location[1]-1]
    left = [location[0]-1, location[1]]
    right = [location[0]+1, location[1]]
    candidates = [up, down, left, right]
    for candidate in candidates:
      if candidate in self.map:
        list_of_possible_coordinates.append(candidate)
    return list_of_possible_coordinates

  # TODOOO
  # add bias to how close the location is to our half since that would probably be more likely

  def updateNoisyDistanceProbabilities(self, mypos, enemy_we_are_checking, gameState):
    index = self.getEnemyListIndex(enemy_we_are_checking)
    distance_to_agents = gameState.getAgentDistances()
    distance_to_enemy = distance_to_agents[enemy_we_are_checking]
    counter = 0
    #Only check possible locations of the enemy in question
    for i in self.emission_probabilties_for_each_location_for_each_agent[index]:
      the_coordinates = tuple(i[0])
      trueDistance = util.manhattanDistance(mypos, the_coordinates)
      emissionModel = gameState.getDistanceProb(trueDistance, distance_to_enemy)
      updated_probabilties_for_each_location = i[1] * emissionModel
      #Todo maybe nromalize all odds after this loop so we arnt left with small numbers
      #find some debugging mechianism
      self.emission_probabilties_for_each_location_for_each_agent[index][counter][1] = updated_probabilties_for_each_location
      counter += 1
      #todo add check for:
      #we know that if the actual distance is equal to 5 or less we always get it as a true reading
      #Get previous location odds and multiply with this information
      #keep track of previous probabilities
    #get most probable location maybe get a couple lets see how only picking the top dog works
    #mostProbableLocation = np.argmax(self.emission_probabilties_for_each_location_for_each_agent[index])

  def get_most_likely_distance_from_noisy_reading(self, enemy):
    index = self.getEnemyListIndex(enemy)
    #just return the likliest location of some agent
    listCopy = [i[1] for i in self.emission_probabilties_for_each_location_for_each_agent[index]]
    max_index = np.argmax(listCopy)
    return tuple(self.emission_probabilties_for_each_location_for_each_agent[index][max_index][0])
    #return max_index

  def reset_agent_probabilties_when_we_know_the_true_position(self, enemy, true_position):
      index = self.getEnemyListIndex(enemy)
      listCopy = [i[0] for i in self.emission_probabilties_for_each_location_for_each_agent[index]]
      location_index = listCopy.index(true_position)
      print('this works?')
      self.emission_probabilties_for_each_location_for_each_agent[index][location_index][1] = 1.0
      print(self.emission_probabilties_for_each_location_for_each_agent[index][location_index])
#  def find_if_we_recently_killed_an_enemy(self,):
