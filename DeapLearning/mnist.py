import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
from readmnist import DataUtils

trainfile_X = 'train-images.idx3-ubyte'
trainfile_y = 'train-labels.idx1-ubyte'
testfile_X = 't10k-images.idx3-ubyte'
testfile_y = 't10k-labels.idx1-ubyte'
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()
# plt.imshow(np.reshape(train_X[2000,:],[28,28]))
# plt.show()

Xp = tf.placeholder(tf.float32, shape=[None, 784])
Yp = tf.placeholder(tf.float32, shape=[None, 10])
Xp1 = tf.reshape(Xp, shape=[-1, 28, 28, 1])


# /*卷积*/
# 定义函数，下面的是方法体
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bais_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


Wconv1 = weight_variable([5, 5, 1, 64])
Bconv1 = bais_variable([64])
# 第一次卷积
conv1 = tf.nn.relu(tf.nn.conv2d(Xp1, Wconv1, strides=[1, 1, 1, 1], padding='VALID') + Bconv1)
# 第一 次池化
pool2 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

Wconv3 = weight_variable([5, 5, 64, 128])
Bconv3 = bais_variable([128])
# 第二次卷积
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, Wconv3, strides=[1, 1, 1, 1], padding='VALID') + Bconv3)
# 第二次池化
pool4 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 拉伸成一个1*10的矩阵
pool4_line = tf.reshape(pool4, shape=[-1, 2048])
Wfc5 = weight_variable([2048, 512])
Bfc5 = bais_variable([512])
# 用relu进行加维度
fc5 = tf.nn.relu(tf.matmul(pool4_line, Wfc5) + Bfc5)
Wfc6 = weight_variable([512, 128])
Bfc6 = bais_variable([128])
# 用relu进行加维度
fc6 = tf.nn.relu(tf.matmul(fc5, Wfc6) + Bfc6)
Wfc7 = weight_variable([128, 10])
Bfc7 = bais_variable([10])
# 用relu进行加维度
fc7 = tf.matmul(fc6, Wfc7) + Bfc7
OUT = tf.nn.softmax(fc7)

# 均方差损失
loss = tf.reduce_mean(-tf.reduce_sum(Yp * tf.log(tf.clip_by_value(OUT, 1e-10, 1.0)), reduction_indices=[1]))
TrainStep = tf.train.AdamOptimizer(0.01).minimize(loss)


# 定义函数
def getBatch(Batch_num):
    # 建立空数组
    Data = []
    Label = []
    for i in range(Batch_num):
        index = np.random.randint(60000)
        Data.append(train_X[index, :])

        labeltemp = np.zeros([10])
        for j in range(10):
            if j == train_y[index,]:
                labeltemp[j] = 1
        Label.append(labeltemp)

    return np.array(Data), np.array(Label)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
# 迭代
for i in range(4000):
    data, label = getBatch(64)
    error, out, result = sess.run([loss, OUT, TrainStep], feed_dict={Xp: data, Yp: label})
    print(error)

print(OUT)
print(train_y.shape)
