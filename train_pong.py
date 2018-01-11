import numpy as np
import gym
import tensorflow as tf

def discount_rewards(rewards, gamma):
    return np.array([sum([gamma**t*r for t, r in enumerate(rewards[i:])]) for i in range(len(rewards))])

def prepro(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

n_features = 80 * 80  # after preprocessing, input is a 1d 6400 dim vector
n_hidden = 200
n_actions = 3  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
learning_rate = 1e-3
gamma = 0.99  # how much does reward propagae backwards

tf.reset_default_graph()

X_ = tf.placeholder(dtype=tf.float32, shape=[None, n_features],name="tf_x")
a_ = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="tf_y")
r_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

w1 = tf.Variable(tf.random_uniform([n_features, n_hidden]))
w2 = tf.Variable(tf.random_uniform([n_hidden, n_actions]))
b1 = tf.Variable(tf.random_uniform([n_hidden]))
b2 = tf.Variable(tf.random_uniform([n_actions]))

z1 = tf.matmul(X_, w1) + b1
fc1 = tf.nn.relu(z1)
z2 = tf.matmul(fc1, w2) + b2
prob = tf.nn.softmax(z2)

loss = tf.losses.log_loss(labels=a_, predictions=prob, weights=r_)
op = tf.train.AdamOptimizer(1e-3).minimize(loss)


env = gym.make("Pong-v0")
save_path='models/pong.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    episode = 0
    
    saver = tf.train.Saver(tf.global_variables())
    try:
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)
    except:
        print("no saved model to load. starting new session")
    else:
        print("loaded model: {}".format(load_path))
        saver = tf.train.Saver(tf.global_variables())
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

        aprob = sess.run(prob, feed_dict={X_: np.reshape(x, (1,-1))})[0]
        a = np.random.choice(n_actions, p=aprob)
        a1e = np.eye(n_actions)[a]
        
        s2, r, d, _ = env.step(a + 1)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        s1 = s2

        # lets ramp up positive rewards for a big gradient boost
        # if r > 0:
            # r = 100

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
            sess.run(op, feed_dict={X_: eps, r_: epr, a_: epl})
            
            # save model
            if episode % 2 == 0:
                print ('ep {}: reward: {}'.format(episode, total_r_per_episode))
            if episode % 50 == 0:
                saver.save(sess, save_path, global_step=episode)
                print("SAVED MODEL #{}".format(episode))
                
            # bookkeeping
            s1 = env.reset()
            xs,rs,ys = [],[],[]
            episode += 1
            total_r_per_episode = 0