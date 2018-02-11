import numpy as np
import tensorflow as tf

from utils import discount_rewards, prepro


n_features = 80 * 80  # after preprocessing, input is a 1d 6400 dim vector
n_hidden = 200
n_actions = 3  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
learning_rate = 1e-3
gamma = 0.99  # how much does reward propagae backwards

class Trainer(object):

    def __init__(self, sess, pong_model, env):
        sess.run(tf.global_variables_initializer())

        episode = 0
    
        saver = tf.train.Saver()
        try:
            ckpt_folder = tf.train.get_checkpoint_state("models")
            load_path = ckpt_folder.model_checkpoint_path
            saver.restore(sess, load_path)
        except:
            print("no saved model to load. starting new session")
        else:
            print("loaded model: {}".format(load_path))
            episode = int(load_path.split('-')[-1])
        
        s1 = env.reset()
        prev_x = None
        xs = []
        rs = []
        ys = []
        total_r_per_episode = 0

        while True:
            env.render()
            cur_x = prepro(s1)
            x = cur_x - prev_x if prev_x is not None else np.zeros(n_features)
            prev_x = cur_x

            aprob = sess.run(pong_model.prob, feed_dict={pong_model.X_: np.reshape(x, (1,-1))})[0]
            a = np.random.choice(n_actions, p=aprob)
            a1e = np.eye(n_actions)[a]
            
            s2, r, d, _ = env.step(a + 1)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            s1 = s2

            # record game history
            xs.append(x)
            ys.append(a1e)
            rs.append(r)
            total_r_per_episode += r

            # done is true when player gets 21 points
            if d:
                
                epr = np.vstack(discount_rewards(rs, gamma))
                eps = np.vstack(xs)
                epl = np.vstack(ys)

                # convert rewards to gaussian distribution - speeds up learning
                epr -= np.mean(epr)
                epr /= np.std(epr)
                
                # train model
                sess.run(pong_model.op, feed_dict={pong_model.X_: eps, pong_model.r_: epr, pong_model.a_: epl})
                
                # save model
                if episode % 1 == 0:
                    print ('ep {}: reward: {}'.format(episode, total_r_per_episode))
                    save_path = 'models/pong.ckpt'
                    pong_model.saver.save(sess, save_path, global_step=episode)
                    print("saved model #{}".format(episode))
                    
                # bookkeeping
                s1 = env.reset()
                xs,rs,ys = [],[],[]
                episode += 1
                total_r_per_episode = 0