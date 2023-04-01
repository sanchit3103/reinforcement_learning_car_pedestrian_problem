import numpy as np
import numpy.random as random

# Initialize environments, separately for car and pedestrian
env             = np.zeros((7, 2))
ped_rows        = 2
ped_cols        = 8
# env[0,0] = 1

# Initialize constants
gamma           = 0.9
alpha           = 0.1
eps             = 0.2
loop_itr        = 0
num_iterations  = 500000

# Initialize actions
actions = []
actions.append(np.array([1,0]))     # Move downwards by 1 step
actions.append(np.array([2,0]))     # Move downwards by 2 steps
actions.append(np.array([1,-1]))    # Move diagonally left
actions.append(np.array([1,1]))     # Move diagonally right

# Formulate state-action dictionary and Q-values
state_action_dict   = {}
Q_values            = {}

for i in range(env.shape[0]):
    for j in range(env.shape[1]):

        _valid_actions_list = []

        for _action in actions:
            index   = tuple([i,j] + _action)
            try:
                env[index]
                valid_index = True

            except (ValueError, IndexError):
                valid_index = False

            if valid_index and index[1] >= 0:
                _valid_actions_list.append(_action)

            # Form action list for terminal states
            elif i == env.shape[0] - 1:
                _valid_actions_list.append(_action)

        # Add pedestrian positions in the state_action dictionary and Q_values dictionary
        for k in range(ped_cols):
            for l in range(ped_cols):
                state_action_dict[((i,j), (0, k), (1, l))]          = _valid_actions_list # state_action_dict[(car_pos, ped1_pos, ped2_pos)] = valid_actions_list
                for _action in _valid_actions_list:
                    Q_values[(((i,j), (0, k), (1, l)), tuple(_action))] = 0

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

def get_reward(_state: tuple, _action: list) -> int:

    # Input:
    # _state    : tuple of a valid state in env
    # _action   : Action to be taken by the car

    # Output:
    # _reward   : Reward according to the car and pedestrian position

    _target_state   = env.shape[0] - 1

    # Transform car and pedestrian coordinates
    car_pos       = tuple(np.array(_state[0]) + np.array([0,3]))
    ped1_pos      = tuple(np.array(_state[1]) + np.array([3,0]))
    ped2_pos      = tuple(np.array(_state[2]) + np.array([3,0]))

    if _state[0][0] != _target_state and car_pos != ped1_pos and car_pos != ped2_pos:
        _reward = round(1 / (_target_state - _state[0][0]), 2)


    elif car_pos == ped1_pos or car_pos == ped2_pos:
        _reward = -10

    elif tuple(_action) == (2, 0) and _state[0][0] != _target_state:
        temp_car_pos    = tuple(np.array(_state[0]) + np.array([0,3] + np.array([-1,0])))
        if temp_car_pos == ped1_pos or temp_car_pos == ped2_pos:
            _reward = -10

    else:
        _reward = 10

    return _reward

def check_terminal_state(_state: tuple, _action: list) -> bool:

    # Input:
    # _state    : tuple of a valid state in env
    # _action   : Action to be taken by the car

    # Output:
    # Boolean as per the terminal state checks

    _target_state   = env.shape[0] - 1

    # Transform car and pedestrian coordinates
    car_pos       = tuple(np.array(_state[0]) + np.array([0,3]))
    ped1_pos      = tuple(np.array(_state[1]) + np.array([3,0]))
    ped2_pos      = tuple(np.array(_state[2]) + np.array([3,0]))

    if car_pos == ped1_pos or car_pos == ped2_pos:
        return True

    elif tuple(_action) == (2, 0) and _state[0][0] != _target_state:
        temp_car_pos    = tuple(np.array(_state[0]) + np.array([0,3] + np.array([-1,0])))
        if temp_car_pos == ped1_pos or temp_car_pos == ped2_pos:
            return True

    elif _state[0][0] == _target_state:
        return True

    else:
        return False


# Q-Learning Loop
while(loop_itr < num_iterations):
    car_state       = (0, random.choice(env.shape[1]))
    ped1_state      = (0, random.choice(ped_cols))
    ped2_state      = (1, random.choice(ped_cols))
    state           = (car_state, ped1_state, ped2_state)
    action          = np.array([1,0])
    env[car_state]  = 1
    loop_itr        += 1
    terminal_state  = False
    if loop_itr % 100000 == 0:
        print('Iteration: ', loop_itr)

    while len(car_state) != 0:

        terminal_state      = check_terminal_state(state, action)

        # Execute the following code if car state is not a terminal state
        if not terminal_state:
            prev_action     = action
            action          = pick_action(state, eps)
            next_car_state  = tuple(np.array(state[0]) + action)
            next_ped1_state = tuple(np.array(state[1]) + np.array([0, -1]))
            next_ped2_state = tuple(np.array(state[2]) + np.array([0, 1]))

            # Checks to prevent pedestrians going out of bounds of environment
            if next_ped1_state[1] < 0:
                next_ped1_state = (0, 0)

            if next_ped2_state[1] > 7:
                next_ped2_state = (1, 7)

            next_state          = (next_car_state, next_ped1_state, next_ped2_state)
            env[car_state]      = 0
            env[next_car_state] = 1

            # Find the max Q-value for next state
            next_state_Q_value_list             = []

            for _action in state_action_dict[next_state]:
                next_state_Q_value_list.append(Q_values[(next_state, tuple(_action))])

            max_Q_next_state                    = np.max(next_state_Q_value_list)

            term1                               = get_reward(state, prev_action) + (gamma * max_Q_next_state)
            Q_values[(state, tuple(action))]    = Q_values[(state, tuple(action))] + alpha * ( term1 - Q_values[(state, tuple(action))] )

            state                               = next_state

        # Terminal state condition
        else:
            prev_action                         = action
            action                              = pick_action(state, eps)
            Q_values[(state, tuple(action))]    = Q_values[(state, tuple(action))] + alpha * ( get_reward(state, prev_action) - Q_values[(state, tuple(action))] )
            car_state                           = ()


for _key, _val in Q_values.items():
    print('s-a: {} and q-value: {}'.format(_key, _val))


# state = (4, 0), (0, 4), (1, 3)
# print(get_reward(state))
# print(env)
# print(np.argwhere(env == 1)[0])
