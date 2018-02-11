import gym
import tensorflow as tf
from Policy_Gradient import Model
from Trainer import Trainer

def main():
    pong_tf_model = Model()
    sess = tf.Session()
    pong_env = gym.make("Pong-v0")
    pong_trainer = Trainer(sess, pong_tf_model, pong_env)

if __name__ == '__main__':
    main()