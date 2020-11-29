from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import tensorflow as tf

strings = tf.Variable(["Hello world"], tf.string)
floats = tf.Variable([3.5412, 2.7762], tf.float64)
ints = tf.Variable([1,2,3], tf.int32)
complexs = tf.Variable([25.9 - 7.33j, 1.23- 4.90j], tf.complex128)

tf.Variable(tf.constant(4.2, shape=[3,3]))

v = tf.Variable(0.0)
v = v+1
print(type(v))

# v.assign_add(1)
# print(v)

# v.assign_sub(1)
# print(v)


x = tf.constant([
            [1,2, 3],
            [4,5,6],
            [7,8,9]
])
print(x)
print("dtype : " , x.dtype)
print("shape : ", x.shape)


x.numpy()


x = tf.constant([
            [1,2, 3],
            [4,5,6],
            [7,8,9]
], dtype=tf.float32)
print(x)
print("dtype : " , x.dtype)


coeffn = np.arrange(26)

shape1= [1,2]
shape2= [4,2]
shape3= [2,2,2,2]


a = tf.constant(coeffn, shape=shape1)
print("\n a:\n", a)

b = tf.constant(coeffn, shape=shape2)
print("\n a:\n", b)

c = tf.constant(coeffn, shape=shape3)
print("\n a:\n", c)


t = tf.constant(np.arrange(80), shape=[5,2,8])

rank = tf.rank(t)
print("rank: ", rank)

t2 = tf.reshape(t, [8,10])

print("t2.shape", t2.shape)

ones = tf.ones(shape=(2,3))
zeros = tf.zeros(shape=(2,4))
eye = tf.eye(3)
tensor7  = tf.constant(7.0, shape=[2,2])
print("\n ones : \n",ones)
print("\n zeros : \n",zeros)
print("\n eye : \n",eye)
print("\n tensor7 : \n",tensor7)


t1 = tf.ones(shape =(2,2))
t2 = tf.zeros(shape=(2,2))

conacat0 = tf.concat([t1,t2], 0)
concat1 = tf.concat([t1,t2], 1)

t = tf.constant(np.arrange(24), shape=[3,3,4])
print("\n t shape:" , t.shape)

t1 = tf.expand_dims(t, 0)
t2 = tf.expand_dims(t, 1)
t3 = tf.expand_dims(t,3)

t1 = tf.squeeze(t1, 0)
t2 = tf.squeeze(t2, 1)
t3 = tf.squeeze(t3, 3)

x = tf.constant([1,2,3,4,5,6,7])

c = tf.constant([[1.0, 2.4],
                [3.0, 4.0]])
d = tf.constant([
            [1.0, 1.0],
            [0.0, 1.0]
])


matmul_cd = tf.matmul(c,d)
print(matmul_cd)

c_times_d = c*d
c_plus_d = c+d
c_minus_d = c-d
c_div_c = c/c

print(c_times_d)
print(c_plus_d)
print(c_minus_d)
print(c_div_c)


a = tf.constant([
        [2,3],
            [3,3]
 ])
b = tf.constant([

        [8,3], 
        [2,3]
 ])

c = tf.constant([

        [-6.09 + 1.76],
        [-2.54+ 2.15]

])

absx = tf.abs(x)

powaa = tf.pow(a, a)
print("\n", absx)
print("\n", powaa)


tn = tf.random.normal(shape=[2,2], mean=0, stddev=1.)
tu = tf.random.uniform(shape=[2,1], minval=0, maxval=10)

tp = tf.random.poison((2,2), 5)

d = tf.square(tn)
e = tf.exp(d)
f = tf.cos(c )