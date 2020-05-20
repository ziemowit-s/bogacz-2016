import numpy as np
import matplotlib.pyplot as plt

from model import Model

if __name__ == '__main__':
    mean = 10
    std = 2

    for epoch in range(50):
        model = Model(num_of_actions=1, actor_only=True)

        g = []
        n = []
        N = [i for i in range(100)]
        for i in N:
            reward = np.random.normal(loc=mean, scale=std, size=1)[0]
            model.act(reward)

            ac = model._actions[0]
            g.append(ac._g)
            n.append(ac._n)

        plt.title("For mean=%s and std=%s" % (mean, std))
        plt.plot(N, g)
        plt.plot(N, n)
    plt.show()
