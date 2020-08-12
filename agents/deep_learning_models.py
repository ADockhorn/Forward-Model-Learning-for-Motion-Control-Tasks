from rl.agents import DQNAgent, SARSAAgent, CEMAgent, NAFAgent
from rl.memory import EpisodeParameterMemory, SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.core import Processor

from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Flatten, Dense, Activation, Input, Concatenate
from tensorflow.python.keras.optimizer_v2.adam import Adam

from rl.random import OrnsteinUhlenbeckProcess

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def init_dqn(env, nb_actions):
    """ Initialize the DQN agent using the keras-rl package.

    :param env: the environment to be played, required to determine the input size
    :param nb_actions: number of actions
    :return: DQN Agent
    """
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # compile agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.model_name = f"DQN"
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def init_sarsa(env, nb_actions, lr=1e-3):
    """ Initialize the Sarsa agent using the keras-rl package.

    :param env: the environment to be played, required to determine the input size
    :param nb_actions: number of actions
    :param lr: learning rate
    :return: Sarsa Agent
    """
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    # SARSA does not require a memory.
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.model_name = f"SARSA"
    sarsa.compile(Adam(lr=lr), metrics=['mae'])
    return sarsa


def init_cem(env, nb_actions):
    """ Initialize the CEM agent using the keras-rl package.

    :param env: the environment to be played, required to determine the input size
    :param nb_actions: number of actions
    :return: CEM agent
    """

    # Option 2: deep network
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))

    # compile agent
    memory = EpisodeParameterMemory(limit=1000, window_length=1)

    cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
                   batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
    cem.model_name = "CEM"
    cem.compile()
    return cem


class PendulumProcessor(Processor):
    """ Modify reward to suit the conditions of NAF
    """
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.


def init_naf(env, nb_actions):
    """ Initialize the NAF agent using the keras-rl package.

    :param env: the environment to be played, required to determine the input size
    :param nb_actions: number of actions
    :return: NAF agent
    """
    # Build all necessary models: V, mu, and L networks.
    v_model = Sequential()
    v_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    v_model.add(Dense(16))
    v_model.add(Activation('relu'))
    v_model.add(Dense(16))
    v_model.add(Activation('relu'))
    v_model.add(Dense(16))
    v_model.add(Activation('relu'))
    v_model.add(Dense(1))
    v_model.add(Activation('linear'))

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation('linear'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    x = Concatenate()([action_input, Flatten()(observation_input)])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
    x = Activation('linear')(x)
    l_model = Model(inputs=[action_input, observation_input], outputs=x)

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    processor = PendulumProcessor()
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = NAFAgent(nb_actions=nb_actions, V_model=v_model, L_model=l_model, mu_model=mu_model,
                     memory=memory, nb_steps_warmup=100, random_process=random_process,
                     gamma=.99, target_model_update=1e-3, processor=processor)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    agent.model_name = "NAF"
    return agent
