import numpy as np
import gym
import tensorflow as tf

# hyperparameters
n_obs = 80 * 80           # dimensionality of observations
n_classes = 6             # number of available actions
learning_rate = 1e-3
gamma = .99               # discount factor for reward
decay = 0.99              # decay rate for RMSProp gradients

# tf operations
def tf_discount_rewards(tf_r):
    discount_f = lambda a, v: a*gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
    return tf_discounted_r

# downsampling
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float32).ravel()

tf.reset_default_graph()

X_raw = tf.placeholder(tf.float64, [None, 6400], name="input")
X_ = tf.cast(tf.reshape(X_raw, (-1, 80, 80, 1)), tf.float32)
Y_ = tf.placeholder(tf.float32, [None, n_classes], name="action")

# tf reward processing (need tf_discounted_epr for policy gradient wizardry)
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")
tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
tf_discounted_epr -= tf_mean
tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)


K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

# 5x5 patch, 1 input channel (because input image has 1 channel), K output channels
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  
B1 = tf.Variable(tf.ones([K])/10)

# 5x5 patch, K input channel, L output channels
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)

# 5x5 patch, L input channel, M output channels
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

# w4 is 20x20xM bc Y3 output is 20x20 with M channels
W4 = tf.Variable(tf.truncated_normal([20 * 20 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, n_classes], stddev=0.1))
B5 = tf.Variable(tf.ones([n_classes])/n_classes)

stride = 1  
Y1 = tf.nn.relu(tf.nn.conv2d(X_, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)  # output is 28x28, stride refers to pixels skipped per convolution

stride = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)  # output is 40x40
stride = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)  # output is 20x20

# since Y3 is 7x7 with M channels, we want to reshape Y3 to YY of shape (-1, 20 * 20 * M)
# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 20 * 20 * M])

# now that this is a flat 1d vector, we can feed it to a vanilla function
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

loss = tf.losses.log_loss(labels=Y_, predictions=Y, weights=tf_discounted_epr)
op = tf.train.AdamOptimizer(0.0001).minimize(loss)

# gamespace 
env = gym.make("Pong-v0") # environment info
observation = env.reset()
prev_x = None
xs,rs,ys = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
save_path='models/pong.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #####
    
    saver = tf.train.Saver(tf.global_variables())
    load_was_success = True # yes, I'm being optimistic
    try:
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)
    except:
        print ("no saved model to load. starting new session")
        load_was_success = False
    else:
        print ("loaded model: {}".format(load_path))
        saver = tf.train.Saver(tf.global_variables())
        episode_number = int(load_path.split('-')[-1])
    
    #####
    
    while True:
        env.render()

        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
        prev_x = cur_x

        # stochastically sample a policy from the network
        aprob = sess.run(Y, feed_dict={X_raw: np.reshape(x, (1,-1))})[0]
        action = np.random.choice(n_classes, p=aprob)
        label = np.zeros_like(aprob)
        label[action] = 1

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # record game history
        xs.append(x)
        ys.append(label)
        rs.append(reward)
        
        

        if done:
            # update running reward
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

            # parameter update
            rs_float = np.array(rs).astype("float32")
            _ = sess.run(op, feed_dict={X_raw: np.vstack(xs), tf_epr: np.vstack(rs_float), Y_: np.vstack(ys)})

            # print progress console
            if episode_number % 10 == 0:
                print ('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))
            else:
                print ('\tep {}: reward: {}'.format(episode_number, reward_sum))
                
                
            # bookkeeping
            xs,rs,ys = [],[],[] # reset game history
            episode_number += 1 # the Next Episode
            observation = env.reset() # reset env
            reward_sum = 0
            
            if episode_number % 50 == 0:
                saver.save(sess, save_path, global_step=episode_number)
                print ("SAVED MODEL #{}".format(episode_number))