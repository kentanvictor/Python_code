# for i in range(10):
#   print(i)
import numpy as np
import tensorflow as tf
# X 为占位符：大小，后面来填的
Xp = tf.placeholder(tf.float32,shape=[4,2])
# Y 为占位符：大小，后面来填的
Yp = tf.placeholder(tf.float32,shape=[4,1])
# 定义函数，下面的是方法体
def weight_variable(shape):
    tw =  tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(tw)
def bais_variable(shape):
    tb = tf.constant(0.1,shape=shape)
    return tf.Variable(tb)
W1=weight_variable([2,3])
B1=bais_variable([3])
L1=tf.matmul(Xp,W1)+B1
# 激活函数：使得成为一个非线性函数。
L1=tf.nn.sigmoid(L1)
W2=weight_variable([3,1])
B2=bais_variable([1])
OUT=tf.matmul(L1,W2)+B2

print(OUT)

data=np.array([[0,0],[1,0],[0,1],[1,1]])
label=np.array([[0],[1],[1],[0]])
# 均方差损失
loss=tf.reduce_mean(tf.square(Yp-OUT))
TrainStep=tf.train.AdamOptimizer(0.1).minimize(loss)

sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
# 迭代
for i in range(4000):
    error,out,result=sess.run([loss,OUT,TrainStep],feed_dict={Xp:data,Yp:label})
    print(error,out)