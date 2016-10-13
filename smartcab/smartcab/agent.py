import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import itertools
import numpy as np

# gets the appropriate action to perform
def get_action(qt, state, epsilon = .2, actions = [None, 'forward', 'right', 'left']):
    if (random.random() < .2): return random.choice(actions) # random action
    else: # select max by greedy, random if tie
        np_array = np.array([qt[(state, action)] for action in actions]) # get q_values for this state
        idx = random.choice(list(np.where(np_array == np_array.max())[0])) # choose random action 
        return actions[idx]

# helper function to get max q_value for a particular state
def get_max(qt, state, actions = [None, 'forward', 'right', 'left']):
    return max([qt[(state, action)] for action in actions])

# initializes the Q-table
def init_qt():
    qt = {}
    waypoints = ['forward', 'right', 'left']
    lights = ['red', 'green']
    traffic = ['forward', 'left', 'right', None]
    actions = [None, 'forward', 'right', 'left']
    states = list(itertools.product(waypoints, lights, traffic, traffic))
    for i in states: qt.update({(i, action): 0 for action in actions})
    return qt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qt = init_qt()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # self.q_table = {}

    def update(self, t):
        alpha = .1; gamma = 0.8; epsilon = .2
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # first, set current to previous
        if (not hasattr(self, 'state')): self.state = None
        if (not hasattr(self, 'action')): self.action = None
        if (not hasattr(self, 'reward')): self.reward = None
        self.prev_state = self.state
        self.prev_action = self.action
        self.prev_reward = self.reward
        # update current
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        
        # TODO: Select action according to your policy
        # action = random.choice(Environment.valid_actions) # step 1 - Implement a Basic Driving Agent
        # action = self.next_waypoint # step 2 - Inform the Driving Agent
        self.action = get_action(self.qt, self.state, epsilon) # step 3 - Implement a Q-Learning Driving Agent

        # Execute action and get reward
        self.reward = self.env.act(self, self.action)

        # TODO: Learn policy based on state, action, reward
        if (self.prev_state is not None): # not first step
            self.qt[(self.prev_state, self.prev_action)] = (1-alpha) * self.qt[(self.prev_state, self.prev_action)] + alpha * \
                (self.prev_reward + gamma * get_max(self.qt, self.state))

        # print ("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, self.action, self.reward))  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    for i in a.qt.items(): print(str(i).replace('(','').replace(')',''))


if __name__ == '__main__':
    run()
