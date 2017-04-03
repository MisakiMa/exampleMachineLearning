# coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np



def open_with_numpy_loadtxt(filename):
    data1 = np.loadtxt(filename, delimiter=',')
    return data1



x = tf.placeholder("float",[None, 4])
W = tf.Variable(tf.zeros([4, 4]))
matmul2 = tf.Variable(tf.zeros([4, 4]))
b = tf.Variable(tf.zeros([4]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float",[None, 4])

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
loss = tf.reduce_mean(tf.square(y - y_))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#ここは任意のデータ
data = tf.convert_to_tensor(open_with_numpy_loadtxt("RGB.csv"))
labels = tf.convert_to_tensor(open_with_numpy_loadtxt("label.csv"))

testData = tf.convert_to_tensor(open_with_numpy_loadtxt("TestData.csv"))
testLabel = tf.convert_to_tensor(open_with_numpy_loadtxt("TestLabel.csv"))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_xs =  data.eval()
    batch_ys = labels.eval()
    for i in range(10000):
        sess.run(train_step, feed_dict={x: batch_xs ,y_: batch_ys})

    # 結果表示
    print ("学習結果")
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print (sess.run(accuracy, feed_dict={x:  testData.eval(), y_: testLabel.eval()}))

    tensor_float = tf.cast(W.eval(), tf.float64)


    matmul2 = tf.nn.softmax(tf.matmul( testData.eval(),tensor_float.eval()) + b.eval())
    print ("matmaul2:")
    print (matmul2[200:400].eval())

    answer = tf.argmax(y,1)
    print ("バイアスの値")
    print (b.eval())
    print ("重みの値")
    print (W.eval())

    print ("test")

    # (1) テスト用データを1000サンプル取得
    # new_x = testData[0:100].eval()
    # new_y_ = testLabel[0:100].eval()
    #
    # accuracy, new_y = sess.run([train_step, y], feed_dict={x:new_x , y_:new_y_ })
    # accuracy = tf.cast(accuracy, "float")
    # print ("Accuracy (for test data): %6.2f%%" % accuracy)
    # print ("True Label:", np.argmax(new_y_[0:4,], 1))
    # print ("Est Label:", np.argmax(new_y[0:4, ], 1))


#参考サイト
#http://qiita.com/haminiku/items/36982ae65a770565458d
#http://minus9d.hatenablog.com/entry/2016/03/24/233236
#http://developers.gnavi.co.jp/entry/tensorflow-deeplearning-2_1
#https://soarcode.jp/posts/260
#http://qiita.com/4Ui_iUrz1/items/35a8089ab0ebc98061c1
