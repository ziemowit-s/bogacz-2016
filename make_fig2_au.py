import numpy as np
import matplotlib.pyplot as plt

from au.aumodel import AUModel

"""
Recreate a fig. 2 from the paper

Parameters:
MEAN - mean of the reward distribution
STD - standard deviation of the reward distribution
"""
if __name__ == '__main__':
    MEAN = 10
    STD = 2

    for epoch in range(50):
        model = AUModel(num_of_states=1, num_of_actions=1, actor_only=True)
        action = model.states[0].actions[0]

        g = []
        n = []
        N = [i for i in range(100)]
        for i in N:
            reward = np.random.normal(loc=MEAN, scale=STD, size=1)[0]
            model.reward(reward)
            g.append(action._g)
            n.append(action._n)

        plt.title("For normal distribution of the reward with mean=%s and std=%s" % (MEAN, STD))
        plt.xlabel("Steps")
        plt.ylabel("Synaptic weight")
        plt.plot(N, g)
        plt.plot(N, n)
    plt.show()
