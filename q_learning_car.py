import numpy as np
import numpy.random as random

# Initialize environment
env = np.zeros((7, 2))
# env[0,0] = 1

# Initialize constants
gamma           = 0.9
alpha           = 0.1
eps             = 0.2
loop_itr        = 0
num_iterations  = 10000

# Initialize actions
actions = []
actions.append(np.array([1,0]))     # Move downwards by 1 step
actions.append(np.array([2,0]))     # Move downwards by 2 steps
actions.append(np.array([1,-1]))    # Move diagonally left
actions.append(np.array([1,1]))     # Move diagonally right

# Formulate state-action dictionary, state-alpha dictionary and Q-values
state_action_dict   = {}
state_alpha_dict    = {}
Q_values            = {}

for i in range(env.shape[0]):
    for j in range(env.shape[1]):

        state_action_dict[(i,j)] = []
        state_alpha_dict[(i,j)] = 0

        for _action in actions:
            index   = tuple([i,j] + _action)
            try:
                env[index]
                valid_index = True

            except (ValueError, IndexError):
                valid_index = False

            if valid_index and index[1] >= 0:
                state_action_dict[(i,j)].append(_action)
                Q_values[((i,j), tuple(_action))] = 0

            # Q-value for Terminal states
            if i == env.shape[0] - 1:
                state_action_dict[(i,j)].append(_action)
                Q_values[((i,j), tuple(_action))] = 0

def pick_action(_state: tuple, eps: float) -> list:

    # Input:
    # _state    : tuple of a valid state in env
    # eps       : Arbitrary positive and small value

    # Output:
    # _action   : Optimal action basis the Q-value

    _valid_actions = state_action_dict[_state]

    if random.random() < eps:
        _action_index   = random.choice(len(_valid_actions))
        _action         = _valid_actions[_action_index]

        return _action

    else:
        _Q_value_list = []
        for _action in _valid_actions:
            _Q_value_list.append(Q_values[(_state, tuple(_action))])

        _action     = _valid_actions[np.argmax(_Q_value_list)]

        return _action

def get_reward(_state: tuple) -> int:

    # Input:
    # _state    : tuple of a valid state in env

    # Output:
    # _reward   : Reward according to the car and pedestrian position

    _terminal_state = env.shape[0] - 1

    if _state[0] != _terminal_state:
        _reward = round(1 / (_terminal_state - _state[0]), 2)

    else:
        _reward = 10

    return _reward

# Q-Learning Loop
while(loop_itr < num_iterations):
    state           = (0, random.choice(env.shape[1]))
    env[state]      = 1
    loop_itr        += 1
    print('Iteration: ', loop_itr)

    while len(state) != 0:

        # Execute the following code if state is not a terminal state
        if state[0] != env.shape[0] - 1:
            action          = pick_action(state, eps)
            next_state      = tuple(np.array(state) + action)
            # alpha           = 1 / (state_alpha_dict[state] + 1) # Calculate alpha basis number of visits to a state
            env[state]      = 0
            env[next_state] = 1

            # Find the max Q-value for next state
            next_state_Q_value_list             = []

            for _action in state_action_dict[next_state]:
                next_state_Q_value_list.append(Q_values[(next_state, tuple(_action))])

            max_Q_next_state                    = np.max(next_state_Q_value_list)

            term1                               = get_reward(state) + (gamma * max_Q_next_state)
            Q_values[(state, tuple(action))]    = Q_values[(state, tuple(action))] + alpha * ( term1 - Q_values[(state, tuple(action))] )

            state                               = next_state

        # Terminal state condition
        else:
            action                              = pick_action(state, eps)
            Q_values[(state, tuple(action))]    = Q_values[(state, tuple(action))] + alpha * ( get_reward(state) - Q_values[(state, tuple(action))] )
            state                               = ()

        # Update state-alpha dictionary
        if state:
            state_alpha_dict[state] = state_alpha_dict[state] + 1

for _key, _val in Q_values.items():
    print('s-a: {} and q-value: {}'.format(_key, _val))

print(env)
