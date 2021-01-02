
import tensorflow as tf

print(tf.__version__)
import numpy as np
# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
# create tensorflow struct start
Weights = tf.Variable(tf.random_uniform((1,), -1.0, 1.0))
biases = tf.Variable(tf.zeros((1,)))
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
# 创建session
sess = tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases), sess.run(loss))

# import tensorflow as tf
# import numpy as np
# # create data
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1 + 0.3
# # create tensorflow structure
# Weights = tf.Variable(tf.random.uniform((1,), -1.0, 1.0))
# biases = tf.Variable(tf.zeros((1,)))
# loss = lambda: tf.keras.losses.MSE(y_data, Weights * x_data + biases)  # alias: tf.losses.mse
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)  # alias: tf.optimizers.SGD
# for step in range(201):
#     optimizer.minimize(loss, var_list=[Weights, biases])
#     if step % 20 == 0:
#         print("{} step, weights = {}, biases = {}, loss = {}".format(step, Weights.read_value(), biases.read_value(), loss()))  # read_value函数可用numpy替换