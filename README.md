# Generating-samples-using-RL

This project was loosely inspired by a paper in chemistry titled *Optimizing distributions over molecular space. An Objective-Reinforced
Generative Adversarial Network for Inverse-design Chemistry.* 

The major idea is to use an mnist-model as the reward function for an ON-policy Monte Carlo algorithm. As a test, I first tried to generate even-numbers using a basic `num % 2' as the reward. The algorithm managed to learn a correct policy. Some of the predictions are:
19664200, 76459822, 33288062, 49823784, 76459870, 21746104, 33214412, 80045112, 21784644.

Details of the model:

actions = {0, 1}

state-space = {(0, 0), (0,1), ..., (27, 27)}; i and j are in [0, 27].

Reward = mnit-model with accuracy > 0.99; below is a test of this model where 0 and 1 are the only allowed values.
![mnist](Images/mnist.png)

Training on 5000 episodes with an epsilon of 0.8 and a reward of 1 for class-prediction of 2, the algorithm learned a policy. This policy, again with epsilon of 0.8, produced the following images. These images received a class-prediction of 2 by the same mnist-model.  
![generated_mnist](Images/generated_mnist.png)


