import gym

from agents.differential_forward_model import DifferentialForwardModel

import pickle
import os
import tensorflow as tf

from agents.deep_learning_models import init_dqn, init_sarsa, init_cem

tf.compat.v1.disable_eager_execution()


if __name__ == "__main__":
    # Get the environment and extract the number of actions.
    for ENV_NAME in ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']:
        RESULT_FOLDER = "checkpoints"
        if ENV_NAME == "Acrobot-v1":
            env = gym.make(ENV_NAME, cont_reward=True)
        else:
            env = gym.make(ENV_NAME)

        env.reset()
        for i in range(5):
            env.step(env.action_space.sample())
        env.render()
        env.seed(123)

        # initialize models
        if not hasattr(env.action_space, 'shape'):
            nb_actions = env.action_space.n
        else:
            nb_actions = env.action_space.shape[0]

        models = {"Sarsa": init_sarsa(env, nb_actions, 1e-3),
                  "CEM": init_cem(env, nb_actions),
                  "DQN": init_dqn(env, nb_actions),
                  "DFM": DifferentialForwardModel(memory_length=2)
                  }

        training_histories = {}
        testing_histories = {}

        # train models and store training results
        nb_steps = 1000
        for i in range(10):
            if not os.path.exists(f"results\\{ENV_NAME}"):
                os.mkdir(f"results\\{ENV_NAME}")

            trial_idx = len([i for i in os.listdir(f"results\\{ENV_NAME}") if os.path.isdir(f"results\\{ENV_NAME}\\{i}")])
            print("store files using trial_idx", trial_idx)
            os.mkdir(f"results\\{ENV_NAME}\\checkpoints-{trial_idx}")

            for model_name, model in models.items():
                model.result_folder = f"results\\{ENV_NAME}\\checkpoints-{trial_idx}"
                model.environment_name = ENV_NAME
                training_histories[model_name] = model.fit(env, nb_steps=nb_steps, visualize=False, verbose=2,
                                                           store_every_x_steps=10000)
                with open(f"results\\{ENV_NAME}\\checkpoints-{trial_idx}\\training_results_{model_name}", "wb") as file:
                    if isinstance(training_histories[model_name], dict):
                        pickle.dump(training_histories[model_name], file)
                    else:
                        pickle.dump(training_histories[model_name].history, file)
