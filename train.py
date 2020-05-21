
import numpy as np

from model import Model


if __name__ == '__main__':
    accuracy = []
    for epoch in range(1):
        model = Model(num_of_actions=2, actor_only=True)

        N = [i for i in range(1000)]
        r = 0
        for i in N:
            state = np.random.randint(low=0, high=2, size=1)[0]
            actions = model.act(r=r, action_index=state)

            ac = np.argmax(actions)
            if state == ac:
                r = 10
                accuracy.append(1)
            else:
                r = -10
                accuracy.append(0)

            if len(accuracy) == 1000:
                accuracy.pop(0)

            if i % 2 == 0 and i != 0:
                avg = np.average(accuracy)
                print('epoch:', epoch, 'i:', i, 'acc:', avg)
