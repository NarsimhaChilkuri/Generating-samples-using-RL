iimport numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm_notebook as tqdm
from keras.models import load_model

num = 2
model = load_model('mnistCNN.h5')
possible_actions = np.arange(0,2)
action_values = np.zeros((28, 28, 2))
counter = np.ones((28, 28, 2))

def incremental_mean(current_mean, new_value, n):
    new_mean = current_mean + 1/n * (new_value - current_mean)
    return new_mean

def compute_reward(x):
    X = np.reshape(x, (1, 28, 28, 1))
    pred = np.asscalar(model.predict_classes(X))
    prob = model.predict(X)[0][num]
    if pred == num:
        return 1*prob
    else:
        return 0

def generate_episode(epsilon):
    episode = np.zeros((28, 28))
    for i in range(0, 28):
        for j in range(0, 28):
            optimal_action = np.random.binomial(1, 0.5)
            if np.random.binomial(1, epsilon):
                action_value = action_values[i, j, :]
                optimal_action = np.asscalar(np.argmax(action_value))
            episode[i,j] = optimal_action
    return episode

def update_action_values(eps):
    reward = compute_reward(eps)
    for i in range(0, 28):
        for j in range(0, 28):
            k = int(eps[i,j])
            n = counter[i, j, k]
            action_values[i,j,k] = incremental_mean(action_values[i,j,k], reward, n)
            counter[i, j, k] += 1

def run_program(no_of_episodes, epsilon):
    for i in tqdm(range(0, no_of_episodes)):
        eps = generate_episode(epsilon)
        update_action_values(eps)
