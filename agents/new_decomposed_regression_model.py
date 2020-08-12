import gym
import matplotlib.pyplot as plt
import numpy as np
import operator
from tqdm import trange
from agents.rolling_horizon_evolutionary_algorithm import RollingHorizonEvolutionaryAlgorithm
from sklearn.tree import DecisionTreeRegressor
import math


class DecomposedRegressionModel:
    def __init__(self, regressor=DecisionTreeRegressor, memory_length=1, max_steps=500000):
        self.models = []
        self.is_trained = False
        self.memory_length = memory_length
        self.memory_states = []
        self.memory_actions = []
        self.regressor = regressor
        self._action_length = None
        self.models = None
        self.max_steps = max_steps
        self.data_index = 0

        self.data_set_x = None
        self.data_set_y = None
        self._state_length = None

        if memory_length <= 0:
            raise ValueError(f"memory_length needs to be >= 1; memory_length = {memory_length}")

    def init_episode(self, env, start_state):
        self.memory_states = [[*start_state]]
        self._state_length = len(self.memory_states[0])
        self.memory_actions = []
        self._action_length = np.array(env.action_space.sample()).flatten().shape[0]
        if self.data_index == 0:
            self.data_set_x = np.zeros((self.max_steps, (self._state_length + self._action_length) * self.memory_length))
            self.data_set_y = np.zeros((self.max_steps, self._state_length + 1))

        if self.models is None:
            self.models = []
            for i in range(self._state_length):
                self.models.append(self.regressor())
            self.models.append(self.regressor())

    def train(self):
        if self.data_index > 0:
            for i in range(self.data_set_y.shape[1]):
                self.models[i].fit(self.data_set_x[:self.data_index, :], self.data_set_y[:self.data_index, i])
            self.is_trained = True

    def add_observation(self, action, new_state, reward):
        self.memory_actions.append([*action.flatten()])
        self.memory_actions = self.memory_actions[-self.memory_length:]

        if len(self.memory_states) >= self.memory_length:
            # add new line to the data_set
            input_tuple = (*[val for state in self.memory_states for val in state],
                           *[val for action in self.memory_actions for val in action])
            output_tuple = (*tuple([x - y for x, y in zip(self.memory_states[-1], new_state)]), reward)
            self.data_set_x[self.data_index, :] = input_tuple
            self.data_set_y[self.data_index, :] = output_tuple
            self.data_index += 1

        self.memory_states.append([*new_state])

        # throw away state observations that exceed our memory limitation
        self.memory_states = self.memory_states[-self.memory_length:] if self.memory_length > 0 else []

    def _get_data_set(self):
        if len(self.data_set) == 0:
            return None, None

        input_columns = len(next(iter(self.data_set)))
        target_colums = len(next(iter(self.data_set[next(iter(self.data_set))])))

        x = np.empty((len(self.data_set), input_columns))
        y = np.empty((len(self.data_set), target_colums))
        try:
            for row_idx, input_pattern in enumerate(self.data_set):
                x[row_idx, :] = input_pattern
                y[row_idx, :] = max(self.data_set[input_pattern].items(), key=operator.itemgetter(1))[0]
        except Exception:
            print()
        return x, y

    def predict(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model.predict(x))
        return np.array(outputs).transpose()

    def evaluate_rollouts(self, rollouts, memory_obs, memory_actions, discount_matrix=None, return_state_predictions=False):
        observations_per_tick = len(memory_obs[0])
        actions_per_tick = 1 if len(memory_actions) == 0 else len(memory_actions[0])
        pred_rewards = np.zeros((rollouts.shape[0], rollouts.shape[1]//actions_per_tick))

        network_input = np.zeros((rollouts.shape[0],
                                  (actions_per_tick+observations_per_tick)*self.memory_length))
        network_input[:, :(observations_per_tick*self.memory_length)] = \
            np.tile(np.array(memory_obs).flatten(), (rollouts.shape[0], 1))
        network_input[:, (observations_per_tick*self.memory_length):-actions_per_tick] = \
            np.tile(np.array(memory_actions).flatten(), (rollouts.shape[0], 1))

        if return_state_predictions:
            pred_states = np.zeros((rollouts.shape[0], rollouts.shape[1]//actions_per_tick*observations_per_tick))

        for i in range(rollouts.shape[1]//actions_per_tick):
            network_input[:, -actions_per_tick:] = rollouts[:, (i*actions_per_tick):((i+1)*actions_per_tick)]
            pred = self.predict(network_input)

            # calculate predicted state based on the most recent state and the differential
            pred[:, :-1] = network_input[:, (observations_per_tick*(self.memory_length-1)):(observations_per_tick*(self.memory_length))] - pred[:, :-1]

            # roll state information to the left and insert new predicted state to the right
            network_input[:, :(observations_per_tick*(self.memory_length-1))] = \
                network_input[:, observations_per_tick:(observations_per_tick*self.memory_length)]
            network_input[:, (observations_per_tick*(self.memory_length-1)):(observations_per_tick*(self.memory_length))] = pred[:, :-1]

            # roll action information to the left
            network_input[:, (observations_per_tick * self.memory_length):-actions_per_tick] = \
                network_input[:, (observations_per_tick * self.memory_length + actions_per_tick):]

            pred_rewards[:, i] = pred[:, -1]

            if return_state_predictions:
                pred_states[:, (observations_per_tick*i):(observations_per_tick*(i+1))] = pred[:, :-1]

        if discount_matrix is not None:
            pred_rewards = np.multiply(pred_rewards, discount_matrix)

        if return_state_predictions:
            return np.sum(pred_rewards, axis=1), pred_states, pred_rewards
        else:
            return np.sum(pred_rewards, axis=1)


def plot_prediction(model):

    state = env.reset()
    states = [state]
    actions = []
    for i in range(100):
        action = action_generator()
        actions.append(action)
        observation, _, _, _ = env.step(action)
        states.append(observation)

    rollouts = np.array(actions).reshape((1,-1))
    obs = [[x for x in states[0].flatten()] + [x for x in states[1].flatten()] + [x for x in states[2].flatten()]]
    pred_states = [obs]
    obs = np.repeat(np.array(obs).reshape(1, -1), 1, 0)
    for i in range(len(actions)):
        network_input = np.hstack((obs, rollouts[:, i].reshape(1, -1)))
        pred = model.predict(network_input)
        obs = obs - pred[:, :-1]
        pred_states.append(obs.flatten().tolist())

    for i in range(len(states[0])):
        plt.plot(range(len(states)), [x[i] for x in states])
        plt.plot(range(len(states)), [x[i] for x in pred_states])
        plt.show()

    env.close()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    #env = gym.make('Acrobot-v1')
    #env = gym.make('Pendulum-v0')
    #env = gym.make('MountainCar-v0')
    #env = gym.make('BipedalWalker-v2')

    action_generator = env.action_space.sample
    action_type = type(action_generator())

    #nb_actions = env.action_space.n
    nb_actions = np.array(env.action_space.sample()).flatten().shape[0]
    obs_dim = env.observation_space.shape[0]

    memory_length = 0
    agent = RollingHorizonEvolutionaryAlgorithm(rollout_actions_length=40, mutation_probability=0.7, num_evals=200,
                                                memory_length=memory_length, discount=0.9)
    model = DecomposedRegressionModel(memory_length=memory_length)

    repetitions = 10
    moves = []
    rewards = []
    for rep in trange(repetitions):
        prev_state = env.reset()
        agent.init_episode(env)
        model.init_episode(prev_state)

        done = False
        move = 0
        reward = 0

        while not done:
            if not model.is_trained:
                action = action_generator()
                if type(action) is not np.ndarray:
                    action = np.array([action])
            else:
                action = np.array(agent._get_next_action(model, prev_state)).reshape(env.action_space.shape)

            if isinstance(action, np.ndarray) and isinstance(action_type, np.ndarray):
                next_state, points, done, info = env.step(action)
            else:
                next_state, points, done, info = env.step(*action.flatten())

            done = (next_state[0] < -2.4 or next_state[0] > 2.4
                    or next_state[2] > 12 * 2 * math.pi / 360 or next_state[2] < -12 * 2 * math.pi / 360)

            model.add_observation(action, next_state, points if not done else 0)
            prev_state = next_state
            env.render()

            move += 1
            reward += points
        model.train()

        moves.append(move)
        rewards.append(reward)
        #print('Total Moves in repetition {}: {}, \t points {}'.format(rep, move, reward))

        #best_reward = reward

        if len(rewards) % 10 == 0:
            print(f'Average Moves DT RHEA Agent: {np.mean(rewards)}')
            plt.plot(range(len(rewards)), rewards)
            #plt.axhline(200)
            plt.show()
