import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from input_data import Data

data=Data("./data/")
label_length=len(data.labels)
print(label_length)

images=tf.placeholder(tf.float32,[None,62,250,3])
labels=tf.placeholder(tf.int32,[None,5,52])
print('labels=',labels)

conv_1=slim.conv2d(images,32,[5,5],1,padding='SAME')
avg_pool_1=slim.avg_pool2d(conv_1,[2,2],[1,1],padding='SAME')

conv_2=slim.conv2d(avg_pool_1,32,[5,5],1,padding='SAME')
avg_pool_2=slim.avg_pool2d(conv_2,[2,2],[1,1],padding='SAME')

conv_3=slim.conv2d(avg_pool_2,32,[3,3])
avg_pool_3=slim.avg_pool2d(conv_3,[2,2],[1,1])

flatten=slim.flatten(avg_pool_3)

fc1=slim.fully_connected(flatten,1024,activation_fn=None)

out0=slim.fully_connected(fc1,label_length,activation_fn=None)
out1=slim.fully_connected(fc1,label_length,activation_fn=None)
out2=slim.fully_connected(fc1,label_length,activation_fn=None)
out3=slim.fully_connected(fc1,label_length,activation_fn=None)
out4=slim.fully_connected(fc1,label_length,activation_fn=None)
print('out0',out0)

# out0_argmax=tf.expand_dims(tf.argmax(out0,1),1)
# out1_argmax=tf.expand_dims(tf.argmax(out1,1),1)
# out2_argmax=tf.expand_dims(tf.argmax(out2,1),1)
# out3_argmax=tf.expand_dims(tf.argmax(out3,1),1)
# out4_argmax=tf.expand_dims(tf.argmax(out4,1),1)
# print(out0_argmax)

# out_score=tf.concat([out0,out1,out2,out3,out4],axis=1)
# print(out_score)

print('labels[:,0,:]=',labels[:,0,:])
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out0, labels=labels[:,0,:]))
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out1, labels=labels[:,1,:]))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out2, labels=labels[:,2,:]))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out3, labels=labels[:,3,:]))
loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out4, labels=labels[:,4,:]))


loss_list= [loss0, loss1, loss2, loss3,loss4]
loss_sum = tf.reduce_sum(loss_list)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_sum)

''''''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000000):
        images_,labels_=data.next_batch(100,'train')
        _,loss_val=sess.run([train_op,loss_sum],feed_dict={images:images_,labels:labels_})
        print("loss_val =",loss_val)




















