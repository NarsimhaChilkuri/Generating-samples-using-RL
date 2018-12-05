# Generating-samples-using-RL

This project was loosely inspired by a paper in chemistry titled *Optimizing distributions over molecular space. An Objective-Reinforced
Generative Adversarial Network for Inverse-design Chemistry.* 

The major idea is to use an mnist-model as the reward function for an ON-policy Monte Carlo algorithm. As a test, I first tried to generate even-numbers using a basic `num % 2' as the reward. The algorithm managed to learn a correct policy. Some of the predictions are:
19664200, 76459822, 33288062, 49823784, 76459870, 21746104, 33214412, 80045112, 21784644.

![mnist](Images/mnist.png)

![generated_mnist](Images/generated_mnist.png)


