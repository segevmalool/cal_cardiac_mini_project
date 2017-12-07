import tensorflow as tf
import data_gen
import scipy as sci


class TrainNN(object):

    """PVC Classifier for ECG signal."""
    def __init__(self, active_record, window_size=400):
        self.window_size = window_size

        self.database = data_gen.ParseArrhythmiaSignals(self.window_size)

        self.active_record = active_record

        # Initialize tensorflow model
        # Dynamic Parameters
        self.learning_rate = 0.005
        self.reg_rate = 0.001

        # Static Parameters
        self.param_size = 0.01
        self.num_classes = 2
        self.num_hidden_1 = 50
        self.alpha = 0.5        

        # Input Data
        self.X_in = tf.placeholder(tf.float64, [None, self.window_size])
        self.y_in = tf.placeholder(tf.int64, [None, 1])

        # Derived Parameters
        self.m = tf.stack(tf.gather(tf.shape(self.X_in), 0))
        self.y_one_hot = tf.cast(tf.reshape(tf.one_hot(self.y_in, 2, on_value=1, off_value=0),
                                            [self.m, self.num_classes]), tf.float64)
        
        # Model Parameters
        self.W1 = tf.Variable(tf.truncated_normal([self.window_size, self.num_hidden_1],
                                                  mean=0,
                                                  stddev=self.param_size,
                                                  dtype=tf.float64))
        self.b1 = tf.Variable(tf.zeros([1, self.num_hidden_1], dtype=tf.float64))

        self.W2 = tf.Variable(tf.truncated_normal([self.num_hidden_1, self.num_classes],
                                                  mean=0,
                                                  stddev=self.param_size,
                                                  dtype=tf.float64))
        self.b2 = tf.Variable(tf.zeros([1, self.num_classes], dtype=tf.float64))

        # Inference
        self.y_hat = tf.matmul(tf.sigmoid(tf.matmul(self.X_in, self.W1) + self.b1), self.W2) + self.b2
        self.p_hat = tf.sigmoid(self.y_hat)
        self.guess = tf.argmax(self.p_hat, 1)

        # L2 Regularized Error Measure
        self.E = ((1/self.m)*tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_one_hot,
                                                                                   logits=self.y_hat)) +
                  self.reg_rate*(tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2))))
        
        # Gradient computation and local search
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.grad = self.opt.compute_gradients(self.E)
        
        self.train_step = self.opt.apply_gradients(self.grad)
        
        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()

    def reinit_tf_session(self):
        batch_x, batch_y = self.generate_data_batch(str(self.active_record))
        self.sess.run(self.init, feed_dict={self.X_in: batch_x, self.y_in: batch_y})

    def set_active_record(self, new_active_record):
        if new_active_record in self.database.record_numbers:
            self.active_record = new_active_record
        else:
            print('new record invalid')

    # Run stochastic gd for num_iter iterations
    def sgd(self, num_iter):
        train_error_per_iter = []
        val_error_per_iter = []
        for i in range(num_iter):
            # Run SGD iteration
            batch_x_train, batch_y_train = self.generate_data_batch(self.active_record)
            e, _ = self.sess.run([self.E, self.train_step], feed_dict={self.X_in: batch_x_train,
                                                                       self.y_in: batch_y_train})
            train_error_per_iter.append(e)

            # Dynamic parameter updates -- inactive
            # self.update_params(self.learning_rate,self.reg_rate,self.batch_size,self.param_size)

            # Validation
            batch_x_val, batch_y_val = self.generate_data_batch(self.active_record)
            e_val = self.sess.run([self.E], feed_dict={self.X_in: batch_x_val, self.y_in: batch_y_val})
            val_error_per_iter.append(e_val)
        return train_error_per_iter, val_error_per_iter

    # Generate data batch using class parameters
    def generate_data_batch(self, record_num, n=100, channel=0):
        dat = self.database.generate_data_batch(record_num, n, channel)

        x = sci.array(list(map(lambda q: sci.array(q[2]), dat)))
        y = sci.reshape(sci.array(list(map(lambda q: sci.array(q[1]), dat))), [n, 1])
        y = 1.0*(y == 'V')
        return x, y

    def predict_batch(self, n=100):
        batch_x, batch_y = self.generate_data_batch(self.active_record, n)
        return batch_y, self.sess.run(self.guess, feed_dict={self.X_in: batch_x})

    def update_params(self, learning_rate, reg_rate, param_size, num_hidden):
        self.learning_rate = learning_rate
        self.param_size = param_size
        self.reg_rate = reg_rate
        self.num_hidden_1 = num_hidden

    def show_params(self):
        print('Learning Rate: ' + str(self.learning_rate))
        print('Regularization Rate: ' + str(self.reg_rate))
        print('Param Size: ' + str(self.param_size))
        print('Number of Hidden Units: ' + str(self.num_hidden_1))

    def save_session(self):
        pass
