import numpy as np
import gym
from agents.new_decomposed_regression_model import DecomposedRegressionModel
from agents.rolling_horizon_evolutionary_algorithm import RollingHorizonEvolutionaryAlgorithm
from gym.wrappers import Monitor
import pickle


class DifferentialForwardModel:
    def __init__(self, memory_length=3, model=None, agent=None):

        if model is None:
            self.model = DecomposedRegressionModel(memory_length=memory_length)
        else:
            self.model = model

        if agent is None:
            self.agent = RollingHorizonEvolutionaryAlgorithm(rollout_actions_length=40, mutation_probability=0.7,
                                                             num_evals=100, memory_length=memory_length, discount=0.9)
        else:
            self.agent = agent

        return

    def init_episode(self, env, prev_state):
        self.agent.init_episode(env)
        self.model.init_episode(env, prev_state)

    def fit(self, env, nb_steps, visualize=False, verbose=2, train_every_x_tick=100, random_steps=0,
            store_every_x_steps=0):
        history = {'episode_reward': [],
                   'nb_episode_steps': [],
                   'nb_steps': []}
        total_steps = 0
        total_episodes = 1

        prev_state = env.reset()
        self.init_episode(env, prev_state)
        action_type = type(env.action_space.sample())

        episode_steps = 0
        episode_reward = 0

        print(f"Training for {nb_steps} steps...")

        while True:
            if not self.model.is_trained or total_steps < random_steps:
                action = env.action_space.sample()
                if type(action) is not np.ndarray:
                    action = np.array([action])
            else:
                action = np.array(self.agent._get_next_action(self.model, prev_state)).flatten()

            if isinstance(action, np.ndarray) and action_type is np.ndarray:
                next_state, reward, done, info = env.step(action)
            else:
                next_state, reward, done, info = env.step(*action)

            self.model.add_observation(action, next_state, reward if not done else 0)
            prev_state = next_state

            total_steps += 1
            episode_steps += 1
            episode_reward += reward

            if visualize:
                env.render()

            if store_every_x_steps>0 and total_steps % store_every_x_steps == 0 :
                with open(f"results\\checkpoints-{self.idx}\\dfm_pickle_{total_steps}_1.model", "wb") as f:
                    pickle.dump(self, f)

            if total_steps == nb_steps:
                history['episode_reward'].append(episode_reward)
                history['nb_episode_steps'].append(episode_steps)
                history['nb_steps'].append(total_steps)
                if verbose == 2:
                    print(f"{total_steps}/{nb_steps}: episode: {total_episodes}, episode steps: {episode_steps}, "
                          f"episode reward: {episode_reward}")
                self.model.train()
                break

            if done:
                history['episode_reward'].append(episode_reward)
                history['nb_episode_steps'].append(episode_steps)
                history['nb_steps'].append(total_steps)

                if verbose == 2:
                    print(f"{total_steps}/{nb_steps}: episode: {total_episodes}, episode steps: {episode_steps}, "
                          f"episode reward: {episode_reward}")

                episode_steps = 0
                episode_reward = 0

                prev_state = env.reset()
                self.init_episode(env, prev_state)
                total_episodes += 1
                if total_steps >= random_steps:
                    self.model.train()
            else:
                if (train_every_x_tick > 0 and (total_steps % train_every_x_tick) == 0) or total_steps == random_steps:
                    self.model.train()

        return history

    def test(self, env, nb_episodes=5, visualize=True):
        history = {'episode_reward': [],
                   'nb_episode_steps': [],
                   'nb_steps': []}

        episode_reward = 0
        episode_steps = 0
        total_steps = 0

        prev_state = env.reset()
        self.init_episode(env, prev_state)
        action_type = type(env.action_space.sample())

        print(f"Testing for {nb_episodes} episodes ...")

        for episode in range(nb_episodes):
            while True:
                if not self.model.is_trained:
                    action = env.action_space.sample()
                    if type(action) is not np.ndarray:
                        action = np.array([action])
                else:
                    action = np.array(self.agent._get_next_action(self.model, prev_state)).flatten()

                if isinstance(action, np.ndarray) and action_type is np.ndarray:
                    next_state, reward, done, info = env.step(action)
                else:
                    next_state, reward, done, info = env.step(*action)

                if visualize:
                    env.render()

                prev_state = next_state

                episode_steps += 1
                episode_reward += reward
                total_steps += 1

                if done:
                    history['episode_reward'].append(episode_reward)
                    history['nb_episode_steps'].append(episode_steps)
                    history['nb_steps'].append(total_steps)

                    print(f"Episode {episode}: reward: {episode_reward}, steps: {episode_steps}")

                    episode_steps = 0
                    episode_reward = 0

                    prev_state = env.reset()
                    self.init_episode(env, prev_state)
                    break

        return history


if __name__ == "__main__":
    ENV_NAME = 'CartPole-v1'
    #ENV_NAME = 'Pendulum-v0'
    #ENV_NAME = 'Acrobot-v1'
    #ENV_NAME = 'LunarLander-v2'
    #ENV_NAME = 'Swimmer-v2'

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    env.seed(123)
    nb_actions = len(np.array(env.action_space.sample()).flatten())

    memory_length = 2
    from sklearn.tree import DecisionTreeRegressor
    model = DecomposedRegressionModel(memory_length=memory_length, regressor=DecisionTreeRegressor)
    agent = RollingHorizonEvolutionaryAlgorithm(rollout_actions_length=10, mutation_probability=0.7,
                                              num_evals=400, memory_length=memory_length, discount=0.9)
    dfm = DifferentialForwardModel(model=model, memory_length=memory_length, agent=agent)

    # produces a video output of the current game, storing one video per episode
    # rendering needs to be active for this
    env = Monitor(env, './video', force=True, video_callable=lambda k: True)

    # set visualize to True in case you are using a Monitor for video recording
    history_train = dfm.fit(env, nb_steps=5000, random_steps=0, visualize=True, verbose=2, train_every_x_tick=100)

    # Finally, evaluate our algorithm for 5 episodes.
    #history_test = dfm.test(env, nb_episodes=5, visualize=True)
    #env.close()

    import matplotlib.pyplot as plt

    state = env.reset()
    states = [[*state]]
    rewards = []
    actions = []
    for i in range(100):
        action = env.action_space.sample()
        actions.append([*np.array(action).flatten()])
        observation, reward, _, _ = env.step(action)
        states.append([*observation])
        rewards.append(reward)

    rollouts = np.array(actions[(memory_length-1):]).flatten().reshape((1, -1))
    obs = [x for x in states[0:memory_length]]
    pred_return, pred_state, pred_reward = dfm.model.evaluate_rollouts(rollouts, obs,
                                                                       actions[0:(memory_length-1)],
                                                                       return_state_predictions=True)

    orig_state = np.hstack((np.array(states[memory_length:]), np.array(rewards[(memory_length-1):]).reshape((-1, 1))))
    pred_state = np.hstack((pred_state.reshape((-1, len(state))), pred_reward.reshape((-1, 1))))
    differences = orig_state - pred_state

    for i in range(differences.shape[1]):
        #plt.plot(differences[:, i])
        plt.subplot(121)
        plt.plot(orig_state[:, i], label="original value")
        plt.plot(pred_state[:, i], label="predicted value")
        plt.subplot(122)
        plt.plot(differences[:, i])
        plt.axhline(y=0)
        if i == (differences.shape[1]-1):
            plt.suptitle("reward")
        plt.show()

    """
    history_dt = history_test = dfm.test(env, nb_episodes=1, visualize=True)

    from sklearn.linear_model import Ridge
    dfm.model.regressor = Ridge
    dfm.model.train()
    history_ridge = dfm.test(env, nb_episodes=1, visualize=True)

    from sklearn.svm import SVR
    dfm.model.regressor = SVR
    dfm.model.train()
    history_svm = dfm.test(env, nb_episodes=1, visualize=True)


    dfm.model.is_trained = False
    history_random = dfm.test(env, nb_episodes=1, visualize=True)

    from sklearn.tree import DecisionTreeRegressor
    dfm.model.regressor = DecisionTreeRegressor
    dfm.model.train()
    history_dt = dfm.test(env, nb_episodes=5, visualize=False)
    #
    """
    #env.close()