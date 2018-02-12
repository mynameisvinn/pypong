import gym
import tensorflow as tf

from Model import Policy_Gradient
from Trainer import Trainer
from utils import read_config

def main(configuration):
    pong_tf_model = Policy_Gradient(configuration)
    sess = tf.Session()
    pong_env = gym.make("Pong-v0")
    pong_trainer = Trainer(sess, pong_tf_model, pong_env, configuration)

if __name__ == '__main__':
    config = read_config()
    main(config)