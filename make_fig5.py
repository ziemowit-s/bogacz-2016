import numpy as np
import matplotlib.pyplot as plt

from model.model import Model


def compute_risk(epoch, steps, a, b, risk_reward_proba, with_critic, safe_reward, risk_reward):
    """
    :return:
        tuple(probability of risky choise, action values)
    """
    risk_probas = []
    action_values = []
    for epoch_i in range(epoch):
        model = Model(num_of_states=1, num_of_actions=2, with_critic=with_critic,
                      alfa=0.1, beta=0.1)

        epoch_action_values = []
        action_values.append(epoch_action_values)

        for step_i in range(steps):
            actions = model.act(a=a, b=b)

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
            epoch_action_values.append(model.states[0].get_values())

        # write last probability of safe action
        risk_probas.append(actions[1])

    return np.average(risk_probas), action_values


def plot_action_values(action_values, epoch_num, title):
    """
    Plot action values for each action. Action value consists of G and N values.
    :param action_values: 
        action_value obtained from compute_risk() function (second element of the returned tuple)
    :param epoch_num: 
        which epoch you want to plot
    :param title: 
        title of the graph
    """
    epoch = action_values[epoch_num]

    fig, axes = plt.subplots(2, 1)
    ax0, ax1 = axes.flatten()
    fig.suptitle(title, fontsize=16)

    ax0.set_title("Action: Safe Reward")
    ax0.plot([i for i in range(len(epoch))], [a[0].g for a in epoch], label="GO")
    ax0.plot([i for i in range(len(epoch))], [a[0].n for a in epoch], label="NO-GO")
    ax0.legend()

    ax1.set_title("Action: Risky reward")
    ax1.plot([i for i in range(len(epoch))], [a[1].g for a in epoch], label="GO")
    ax1.plot([i for i in range(len(epoch))], [a[1].n for a in epoch], label="NO-GO")
    ax1.legend()


def print_probas(states):
    for i, s in enumerate(states):
        print(i, s)


"""
Train Actor Uncertainty Model with uncertain reward.
"""
if __name__ == '__main__':
    """
    risky lever gave higher expected reward in the 100% and 50% conditions while choosing the safe 
    lever had higher mean reward in the 12.5% condition
    """
    SAFE_REWARD = 1
    RISK_REWARD = 4

    EPOCH = 1
    BATCH_NUM = 1000
    WITH_CRITIC = False
    RISK_REWARD_PROBA = [1.0, 0.5, 0.25, 0.125]

    params = {
        "d1_agonist": {'a': 3.13, 'b': 0.59, 'a_cont': 1.71},
        "d2_agonist": {'a': 2.27, 'b': 0.39, 'b_cont': 1.86},

        "d1_antagonist": {'a': 0.86, 'b': 1.04, 'a_cont': 2.67},
        "d2_antagonist": {'a': 1.95, 'b': 2.16, 'b_cont': 0.04},
    }

    fig, axs = plt.subplots(2, 2)
    axs = np.reshape(axs, [4, 1])
    x_range = [i for i in range(len(RISK_REWARD_PROBA))]

    y_proba = [0, 20, 40, 60, 80, 100]
    y_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for i, (manipulation, p) in enumerate(params.items()):
        ax = axs[i][0]
        control = []
        intervention = []

        for proba in RISK_REWARD_PROBA:
            a = p['a']
            b = p['b']
            risk_proba, action_values = compute_risk(epoch=EPOCH, steps=BATCH_NUM, a=a, b=b,
                                                     risk_reward_proba=proba, with_critic=WITH_CRITIC,
                                                     safe_reward=SAFE_REWARD, risk_reward=RISK_REWARD)
            intervention.append(risk_proba)

            if 'a_cont' in p:
                a = p['a_cont']
            else:
                b = p['b_cont']

            risk_proba, action_values = compute_risk(epoch=EPOCH, steps=BATCH_NUM, a=a, b=b,
                                                     risk_reward_proba=proba, with_critic=WITH_CRITIC,
                                                     safe_reward=SAFE_REWARD, risk_reward=RISK_REWARD)
            control.append(risk_proba)

            # Show GO/NO-GO population only for control D1 agonist 0.5 probability
            if manipulation == "d1_agonist" and proba == 0.5:
                title = f"CONTROL: {manipulation}, Probability: {proba}"
                plot_action_values(action_values, epoch_num=0, title=title)

        ax.set_ylim([0, 1])
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_proba)

        ax.set_xticks(x_range)
        ax.set_xticklabels([100, 50, 25, 12.5])

        ax.plot(x_range, intervention, label=manipulation, color='black', marker='o',
                markersize=5)
        ax.plot(x_range, control, label="control", linestyle='--', color='gray', marker='o',
                markersize=5)

        ax.legend(loc="upper right")
        ax.set_xlabel("Risky lever probability")
        ax.set_ylabel("% choice of Risky lever")

        print(manipulation)
        print('inte:', intervention)
        print('cont:', control)

    plt.show()
