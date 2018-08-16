import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np

class StochasticWassersteinBarycenter(object):
    def __init__(self, distributions, input_dim, num_positions, num_samples=100000, tol=1e2, beta=0.99, alpha=0.001):
        self.distributions = distributions
        self.num_dist = len(self.distributions)
        self.input_dim = input_dim
        self.num_positions = num_positions
        self.num_samples = num_samples
        self.tol = tol
        self.beta = beta
        self.alpha = alpha
        
        self.graph = self.build_graph()       
               
    def build_graph(self):
        
        graph = tf.get_default_graph()
        
        with graph.as_default():

            self.samples = tf.Variable(self.get_samples(self.num_samples))

            self.X = tf.Variable(tfd.Uniform(low=-5.0, high=5.0).sample([self.num_positions, self.input_dim]))
            self.w = tf.Variable(tfd.Uniform(low=-1.0, high=1.0).sample([self.num_dist, self.num_positions]))    
            z = tf.Variable(tf.zeros_like(self.w))

            beta = tf.constant(self.beta)
            alpha = tf.constant(self.alpha)

            grad = self.estimate_weights_partials()
            
            self.z_update = z.assign(beta * z + grad)
            self.w_update = self.w.assign(self.w + alpha*z)
            self.error = tf.Print(tf.norm(grad, axis=1), [tf.norm(grad, axis=1)], message='Gradient errors : ')

            self.samples_update = self.samples.assign(self.get_samples(self.num_samples))    
            self.X_update = self.X.assign(self.estimate_positions_update()) 
        return graph
    
    def fit(self, num_iterations=10):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_iterations):
                while True:
                    _ = sess.run([self.z_update, self.w_update, self.samples_update, self.error])
                    if (_[-1] <= self.tol*np.ones(2)).all():
                        break
                sess.run(self.X_update)
            positions = sess.run(self.X)
        return positions
                              
    def get_samples(self, num_samples):
        return tf.stack([dist.sample(num_samples) for dist in self.distributions]) 
    
    def sqdist(self, X, X2=None):
        Xs = tf.reduce_sum(tf.square(X), axis=2)  
        if X2 is None:
            return -2 * tf.matmul(X, tf.transpose(X, perm=[0, 2, 1])) + \
                    tf.expand_dims(Xs, axis=2) + tf.expand_dims(Xs, axis=1)
        else:
            X2s = tf.reduce_sum(tf.square(X2), axis=2) 
            return -2 * tf.matmul(X, tf.transpose(X2, perm=[0, 2, 1])) + \
                    tf.expand_dims(Xs, axis=2) + tf.expand_dims(X2s, axis=1)
            
    def calculate_distances(self):
        return self.sqdist(self.samples, tf.tile(tf.expand_dims(self.X, axis=0), [tf.shape(self.samples)[0], 1, 1])) - tf.expand_dims(self.w, axis=1)

    def estimate_powercell_densities(self):
        return tf.reduce_mean(tf.one_hot(tf.argmin(self.calculate_distances(), axis=2), depth=tf.shape(self.w)[1]), axis=1)

    def estimate_weights_partials(self):
        return (1 / self.num_dist) * ((1 / self.num_positions) - self.estimate_powercell_densities())
    
    def estimate_powercell_means(self):
        min_distances = tf.one_hot(tf.argmin(self.calculate_distances(), axis=2), depth=tf.shape(self.w)[1], on_value=True, off_value=False)
        broad_samples = tf.tile(tf.expand_dims(self.samples, 2), [1, 1, tf.shape(self.w)[1], 1])
        return tf.reduce_mean(tf.where(tf.tile(tf.expand_dims(min_distances, -1), [1, 1, 1, tf.shape(self.X)[1]]),
                                       broad_samples, tf.zeros_like(broad_samples)), axis=1) #/ tf.expand_dims(tf.reduce_mean(tf.one_hot(min_distances, depth=tf.shape(self.w)[1]), axis=1), -1) 

    def estimate_positions_update(self):
        densities = tf.expand_dims(self.estimate_powercell_densities(), -1)
        means = self.estimate_powercell_means()
        return tf.reduce_sum(means, axis=0) / tf.reduce_sum(densities, axis=0)