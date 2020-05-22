Implementation of the paper: Mikhael, J. G. & Bogacz, R. Learning Reward Uncertainty in the Basal Ganglia. 
PLoS Comput. Biol. (2016). doi:10.1371/journal.pcbi.1005062

## Run

### make_fig2_au.py
Recreate a fig. 2 from the paper

Parameters:
* MEAN - mean of the reward distribution
* STD - standard deviation of the reward distribution

### train_au_with_deterministic_reward.py
Train Actor Uncertainty Model with detrministic reward.

Parameters (in file): 
* EPOCH - number of epoches 
* ACTOR_ONLY - if true model will not use critic and 
* STATE_AND_ACTION_NUM - number of states and actions (same number here)
  * for state=0 action=0 is rewarded, for state=n action=n is rewarded etc.
  * if work well should return Probability matrix with highest probabilities on diagonal,
    example for STATE_AND_ACTION_NUM=4 and BATCH_NUM=100:
      ```
      [0.415126880851062, 0.19495770638297932, 0.19495770638297932, 0.19495770638297932]
      [0.18563139548871202, 0.40773597674796314, 0.2033163138816624, 0.2033163138816624]
      [0.1883143045894067, 0.1883143045894067, 0.41711656975760747, 0.20625482106357915]
      [0.19061311338016393, 0.19061311338016393, 0.19061311338016393, 0.4281606598595082]
    ```