import tensorflow as tf

class Model():
    def __init__(self):
        self.build_model()
        self.init_saver()

    def build_model(self):
        n_features = 80 * 80  # after preprocessing, input is a 1d 6400 dim vector
        n_hidden = 200
        n_actions = 3  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        lr = 1e-3

        self.X_ = tf.placeholder(dtype=tf.float32, shape=[None, n_features],name="tf_x")
        self.a_ = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="tf_y")
        self.r_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

        w1 = tf.Variable(tf.random_uniform([n_features, n_hidden]))
        w2 = tf.Variable(tf.random_uniform([n_hidden, n_actions]))
        b1 = tf.Variable(tf.random_uniform([n_hidden]))
        b2 = tf.Variable(tf.random_uniform([n_actions]))

        z1 = tf.matmul(self.X_, w1) + b1
        fc1 = tf.nn.relu(z1)
        z2 = tf.matmul(fc1, w2) + b2
        self.prob = tf.nn.softmax(z2)

        self.loss = tf.losses.log_loss(labels=self.a_, predictions=self.prob, weights=self.r_)
        self.op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=2)