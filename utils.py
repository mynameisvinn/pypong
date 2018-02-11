import numpy as np

def discount_rewards(rewards, gamma):
    """calculate npv for each state, based on terminal reward"""
    return np.array([sum([gamma**t*r
        for t, r in enumerate(rewards[i:])]) for i in range(len(rewards))])

def prepro(img):
    """preprocess raw features to reduce dimensionality"""
    img = img[35:195] # crop
    img = img[::2, ::2, 0] # downsample by factor of 2
    img[img == 144] = 0  # erase background (background type 1)
    img[img == 109] = 0  # erase background (background type 2)
    img[img != 0] = 1    # everything else (paddles, ball) just set to 1
    return img.astype(np.float).ravel()
