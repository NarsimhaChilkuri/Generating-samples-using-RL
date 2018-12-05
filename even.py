import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from tqdm import tqdm_notebook as tqdm
from keras.models import load_model

num_len = 8
possible_actions = np.arange(0,10)
action_values = np.zeros((10, 10**(num_len -1)))
counter = np.ones((10, 10**(num_len -1)))

def incremental_mean(current_mean, new_value, n):
    new_mean = current_mean + 1/n * (new_value - current_mean)
    return new_mean

def compute_reward(x):
    div = int(x) % 2
    if div == 0:
        return 1
    else:
        return 0

def generate_episode(epsilon):
    episode = ""
    for i in range(0, num_len):
        if i == 0:
            optimal_action = str(np.random.randint(1, 10))
        else:
            optimal_action = str(np.random.randint(0, 10))
            if np.random.binomial(1, epsilon):
                action_value = action_values[:, int(episode)]
                max_value = max(action_value)
                best_action_indices = [j for j, x in enumerate(action_value) if x == max_value]
                optimal_action = best_action_indices[np.random.randint(0, len(best_action_indices))]
        episode += str(optimal_action)
    return episode

def update_action_values(eps):
    reward = compute_reward(eps)
    for i,s in enumerate(eps):
        if i == 0:
            n = counter[int(s), i]
            action_values[int(s), i] = incremental_mean(action_values[int(s), i], reward, n)
            counter[int(s), i] += 1
        else:
            n = counter[int(eps[i:i+1]), int(eps[0:i])]
            action_values[int(eps[i:i+1]), int(eps[0:i])] = incremental_mean(action_values[int(eps[i:i+1]), int(eps[0:i])], reward, n)

def run_program(no_of_episodes, epsilon):
    for i in tqdm(range(0, no_of_episodes)):
        eps = generate_episode(epsilon)
        update_action_values(eps)
