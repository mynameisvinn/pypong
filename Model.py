import tensorflow as tf

class Policy_Gradient(object):

    def __init__(self, config):
        self.config = config
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.X_ = tf.placeholder(dtype=tf.float32, shape=[None, self.config.n_features],name="tf_x")
        self.a_ = tf.placeholder(dtype=tf.float32, shape=[None, self.config.n_actions],name="tf_y")
        self.r_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

        w1 = tf.Variable(tf.random_uniform([self.config.n_features, self.config.n_hidden]))
        w2 = tf.Variable(tf.random_uniform([self.config.n_hidden, self.config.n_actions]))
        b1 = tf.Variable(tf.random_uniform([self.config.n_hidden]))
        b2 = tf.Variable(tf.random_uniform([self.config.n_actions]))

        z1 = tf.matmul(self.X_, w1) + b1
        fc1 = tf.nn.relu(z1)
        z2 = tf.matmul(fc1, w2) + b2
        self.prob = tf.nn.softmax(z2)

        self.loss = tf.losses.log_loss(labels=self.a_, predictions=self.prob, weights=self.r_)
        self.op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=2)