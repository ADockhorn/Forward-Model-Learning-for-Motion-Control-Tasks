import numpy as np
import math


class RollingHorizonEvolutionaryAlgorithm:

    def __init__(self, rollout_actions_length, mutation_probability, num_evals, memory_length, use_shift_buffer=True,
                 flip_at_least_one=True, discount=None):

        self._rollout_actions_length = rollout_actions_length
        self._use_shift_buffer = use_shift_buffer
        self._flip_at_least_one = flip_at_least_one
        self._mutation_probability = mutation_probability
        self._num_evals = num_evals
        self._memory_length = memory_length
        self._memory_states = None
        self._memory_actions = None
        self._action_generator = None

        if memory_length <= 0:
            raise ValueError(f"memory_length needs to be >= 1; memory_length = {memory_length}")

        self._discount = discount
        if self._discount is not None:
            self.discount_matrix = np.tile(np.array([math.pow(self._discount, x) for x in range(self._rollout_actions_length)]), (self._num_evals, 1))
        else:
            self.discount_matrix = None

    def init_episode(self, env):
        self._memory_states = []
        self._memory_actions = []
        self._action_generator = env.action_space.sample
        self._action_length = np.array(env.action_space.sample()).flatten().shape[0]

        # Initialize the solution to a random sequence
        if self._use_shift_buffer:
            self._solution = self._random_solution()

    def _get_next_action(self, model, obs, report_prediction=False):
        """
        Get the next best action by evaluating a bunch of mutated solutions
        """
        self._memory_states.append([*obs])
        self._memory_states = self._memory_states[-self._memory_length:]
        pred_state = None
        pred_reward = None
        best_idx = -1

        if len(self._memory_states) >= self._memory_length:
            # use rhea to determine action
            if self._use_shift_buffer:
                solution = self._shift_and_append(self._solution)
            else:
                solution = self._random_solution()

            candidate_solutions = self._new_mutate(solution, self._mutation_probability)

            mutated_scores, pred_state, pred_reward = model.evaluate_rollouts(candidate_solutions, self._memory_states, self._memory_actions,
                                                        self.discount_matrix, return_state_predictions=True)
            best_idx = np.argmax(mutated_scores, axis=0)

            # best_score_in_evaluations = mutated_scores[best_idx]

            # The next best action is the first action from the solution space
            self._solution = candidate_solutions[best_idx]
            action = self._solution[:self._action_length]
            #print("apply action:", action)
        else:  # use random action since the memory buffer is not filled yet
            action = self._action_generator()

        if type(action) is np.ndarray:
            self._memory_actions.append([*action])
        else:
            self._memory_actions.append([action])

        self._memory_actions = self._memory_actions[-(self._memory_length - 1):] if self._memory_length > 1 else []

        if report_prediction:
            if best_idx > -1:
                return action, pred_state[best_idx, 0:5], pred_reward[best_idx][0]
            else:
                return action, [], []

        else:
            return action

    def _shift_and_append(self, solution):
        """
        Remove the first element and add a random action on the end
        """
        new_solution = np.copy(solution[self._action_length:])
        new_solution = np.hstack([new_solution, self._action_generator()])
        return new_solution

    def _random_solution(self):
        """
        Create a random set fo actions
        """
        if type(self._action_generator()) is np.ndarray:
            return np.concatenate([self._action_generator() for _ in range(self._rollout_actions_length)])
        else:
            return np.array([self._action_generator() for _ in range(self._rollout_actions_length)])

    def _mutate(self, solution, mutation_probability):
        """
        Mutate the solution
        """

        candidate_solutions = []
        # Solution here is 2D of rollout_actions x batch_size
        for b in range(self._num_evals):
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution)
            if self._flip_at_least_one:
                mutation_indexes.add(np.random.randint(solution_length))

            mutation_indexes = mutation_indexes.union(
                set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes)
            mutations = [self._action_generator() for _ in range(num_mutations)]
            if type(mutations[0]) is np.ndarray:
                mutations = np.concatenate(mutations)

            # Replace values in the solutions with mutated values
            new_solution = np.copy(solution)
            new_solution[list(mutation_indexes)] = mutations
            candidate_solutions.append(new_solution)

        return np.stack(candidate_solutions)

    def _new_mutate(self, solution, mutation_probability):
        """
        Mutate the solution
        """

        candidate_solutions = []
        # Solution here is 2D of rollout_actions x batch_size
        for b in range(self._num_evals):
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution) // self._action_length
            if self._flip_at_least_one:
                index = np.random.randint(solution_length)
                mutation_indexes = mutation_indexes.union(
                    set(range(index * self._action_length, index * self._action_length + self._action_length)))

            mutation_indexes = mutation_indexes.union(
                set(np.where(
                    np.repeat(np.random.random([solution_length]), self._action_length) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes) // self._action_length
            mutations = [self._action_generator() for _ in range(num_mutations)]
            if type(mutations[0]) is np.ndarray:
                mutations = np.concatenate(mutations)

            # Replace values in the solutions with mutated values
            new_solution = np.copy(solution)
            new_solution[list(mutation_indexes)] = mutations
            candidate_solutions.append(new_solution)

        return np.stack(candidate_solutions)

