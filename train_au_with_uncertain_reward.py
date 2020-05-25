import numpy as np
import matplotlib.pyplot as plt

from au.aumodel import AUModel


def compute_risk(epoch, steps, a, b, risk_reward_proba, actor_only, safe_reward, risk_reward):
    """
    :return:
        probability of risky choise
    """
    risk_probas = []
    for epoch_i in range(epoch):
        model = AUModel(num_of_states=1, num_of_actions=2, actor_only=actor_only)

        for step_i in range(steps):
            actions = model.act(a=a, b=b)
            #if step_i % 100 == 0:
                #print(actions)

            # compute action: 0 or 1
            if np.random.random(size=1)[0] <= actions[0]:
                # Action 0: Safe reward
                action = 0
                r = safe_reward
            else:
                # Action 1: Risk reward
                action = 1
                # compute risk reward with proba
                if np.random.random(size=1)[0] <= risk_reward_proba:
                    r = risk_reward
                else:
                    r = 0

            model.reward(reward=r, action=action)

        # write last probability of safe action
        risk_probas.append(actions[1])

    return np.average(risk_probas)


def print_probas(states):
    for i, s in enumerate(states):
        print(i, s)


"""
Train Actor Uncertainty Model with uncertain reward.
"""
if __name__ == '__main__':
    SAFE_REWARD = 2
    RISK_REWARD = 4
    RISK_REWARD_PROBA = [1.0, 0.5, 0.25, 0.125]

    EPOCH = 100
    BATCH_NUM = 10000
    ACTOR_ONLY = True

    params = {
        "d1_agonist": {'a': 3.13, 'b': 0.59, 'a_cont': 1.71},
        "d2_agonist": {'a': 2.27, 'b': 0.39, 'b_cont': 1.86},

        "d1_antagonist": {'a': 0.86, 'b': 1.04, 'a_cont': 2.67},
        "d2_antagonist": {'a': 1.95, 'b': 2.16, 'b_cont': 0.04},
    }

    fig, axs = plt.subplots(2, 2)
    axs = np.reshape(axs, [4, 1])

    for i, (manipulation, p) in enumerate(params.items()):
        ax = axs[i][0]
        control = []
        intervention = []

        for proba in RISK_REWARD_PROBA:
            a = p['a']
            b = p['b']
            risk_proba = compute_risk(epoch=EPOCH, steps=BATCH_NUM, a=a, b=b,
                                      risk_reward_proba=proba, actor_only=ACTOR_ONLY,
                                      safe_reward=SAFE_REWARD, risk_reward=RISK_REWARD)
            intervention.append(risk_proba)

            if 'a_cont' in p:
                a = p['a_cont']
            else:
                b = p['b_cont']

            risk_proba = compute_risk(epoch=EPOCH, steps=BATCH_NUM, a=a, b=b,
                                      risk_reward_proba=proba, actor_only=ACTOR_ONLY,
                                      safe_reward=SAFE_REWARD, risk_reward=RISK_REWARD)
            control.append(risk_proba)

        ax.plot([i for i in range(len(RISK_REWARD_PROBA))], intervention, label=manipulation)
        ax.plot([i for i in range(len(RISK_REWARD_PROBA))], control, label="control")

        print(manipulation)
        print('inte:', intervention)
        print('cont:', control)

    fig.legend()
    plt.show()
