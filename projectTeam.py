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
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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

class DecisionTree :
  levels = 4

  def __init__(self,game_state,root_agent_object):
      self.root_node = DecisionTreeNode(game_state,root_agent_object.index,root_agent_object)

      self.add_children(self.root_node, 1)

  def add_children(self,tree_node,level):
    if level > self.levels:
      return

    future_states = []

    if (tree_node.node_state.data.agentStates[tree_node.agent_index].configuration == None):
      future_states.append(tree_node.node_state) # I.E current state
    else:
      actions = tree_node.node_state.getLegalActions(tree_node.agent_index)
      for i in actions:
        future_states.append(tree_node.node_state.generateSuccessor(tree_node.agent_index, i))

    for i in range(len(future_states)):
      future_node_index = (tree_node.agent_index + 1) % 4
      next_node = tree_node.add_child(future_states[i], future_node_index)

      self.add_children(next_node,level + 1)



class DecisionTreeNode :
  def __init__(self,game_state,agent_index,root_agent_object):
    self.node_state = game_state
    self.agent_index = agent_index #The index of the agent that is supposed to move
    self.root_agent_object = root_agent_object
    self.node_score = self.evaluate_state()
    self.children = []


  def add_child(self,game_state,index):
    child_node = DecisionTreeNode(game_state,index,self.root_agent_object)
    self.children.append(child_node)

    return child_node

  def evaluate_state(self):
    if (self.agent_index == self.root_agent_object.index or (self.agent_index + 2) % 4 == self.root_agent_object.index):
      enemy_team = False
    else:
      enemy_team = True

    if (self.node_state.data.agentStates[self.agent_index].configuration != None):
      agent_position = self.node_state.data.agentStates[self.agent_index].configuration.pos
    else:
      agent_position = (1,1)

    distance_to_food_factor = self.get_distances_to_food_factor(agent_position,enemy_team)

    return_with_food_factor = self.get_return_with_food_factor(agent_position,self.agent_index)

    state_value = distance_to_food_factor + return_with_food_factor
    
    if (enemy_team):
      state_value = - state_value

    return state_value

  def get_distances_to_food_factor(self,agent_position,enemy_team):
    if (enemy_team):
      food = CaptureAgent.getFoodYouAreDefending(self.root_agent_object,self.node_state)
    else:
      food = CaptureAgent.getFood(self.root_agent_object, self.node_state)

    distances = []

    for i in range(0,food.width):
      for j in range(0,food.height):
        if food.data[i][j]:
          distance = CaptureAgent.getMazeDistance(self.root_agent_object, agent_position, (i,j))
          distances.append(distance)

    distance_to_food_factor = sum(distances) / float(len(distances))
    distance_to_food_factor = 1 / distance_to_food_factor
    distance_to_food_factor = distance_to_food_factor * 100

    return distance_to_food_factor

  def get_return_with_food_factor(self,agent_position,agent_index):
    food_carrying = self.node_state.data.agentStates[self.agent_index].numCarrying

    if food_carrying == 0:
      return 0

    distance_home = self.get_closest_distance_to_home(agent_position,self.agent_index)

    return_with_food_factor = (1 / distance_home) * food_carrying * 1000

    return return_with_food_factor


  def get_closest_distance_to_home(self,agent_position,agent_index):
    blue_side = agent_index % 2

    width = self.node_state.data.layout.width
    height = self.node_state.data.layout.height

    if blue_side:
      col = int(width/2)
    else:
      col = int(width/2 - 1)


    distances = []

    for i in range(height):
      if not self.node_state.data.layout.isWall((col,i)):
        dist = CaptureAgent.getMazeDistance(self.root_agent_object,agent_position,(col,i))
        distances.append(dist)

    return min(distances)





class DummyAgent(CaptureAgent):
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
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    decision_tree = DecisionTree(gameState,self)

    a = CaptureAgent.getCurrentObservation(self)

    a = CaptureAgent.getFood(self,gameState)

    a = CaptureAgent.getMazeDistance(self, (1, 1), (3, 1))






    a = 2

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

