import numpy as np

from au.aumodel import AUModel

"""
Train Actor Uncertainty Model with detrministic reward.

Choose: 
EPOCH - number of epoches 
ACTOR_ONLY - if true model will not use critic and 
STATE_AND_ACTION_NUM - number of states and actions (same number here)
  * for state=0 action=0 is rewarded, for state=n action=n is rewarded etc.
  * if work well should return Probability matrix with highest probabilities on diagonal,
    example for STATE_AND_ACTION_NUM=4 and BATCH_NUM=100:
      [0.415126880851062, 0.19495770638297932, 0.19495770638297932, 0.19495770638297932]
      [0.18563139548871202, 0.40773597674796314, 0.2033163138816624, 0.2033163138816624]
      [0.1883143045894067, 0.1883143045894067, 0.41711656975760747, 0.20625482106357915]
      [0.19061311338016393, 0.19061311338016393, 0.19061311338016393, 0.4281606598595082]
"""
if __name__ == '__main__':
    EPOCH = 1
    BATCH_NUM = 100
    ACTOR_ONLY = True
    STATE_AND_ACTION_NUM = 4

    accuracy = []
    for epoch in range(EPOCH):
        model = AUModel(num_of_states=STATE_AND_ACTION_NUM,
                        num_of_actions=STATE_AND_ACTION_NUM, actor_only=ACTOR_ONLY)

        N = [i for i in range(BATCH_NUM)]
        r = 0
        action = 0
        for i in N:
            state = np.random.randint(low=0, high=STATE_AND_ACTION_NUM, size=1)[0]
            actions = model.act(state=state, a=1, b=1)
            action = int(np.argmax(actions))

            if state == action:
                r = 1
                accuracy.append(1)
            else:
                r = -1
                accuracy.append(0)
            model.reward(reward=r, state=state, action=action)

            if len(accuracy) == round(len(N)/10):
                accuracy.pop(0)

            avg = np.average(accuracy)
            print('epoch:', epoch, 'i:', i, 'acc:', avg)

        print("Probabilities:")
        for i, s in enumerate(model.states):
            print(i, s)
