import numpy as np

def discount_rewards(rewards, gamma):
    return np.array([sum([gamma**t*r for t, r in enumerate(rewards[i:])]) for i in range(len(rewards))])

def prepro(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()