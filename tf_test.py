import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


data, label = mnist.train.next_batch(10)

print(data.shape, label.shape)



def convnet(inputs, num_classes, droupout=0.5, training=True, reuse=True):
    
    
    with tf.variable_scope('convnet', reuse=reuse):

        
        x = tf.layers.conv2d(inputs, 64, 5, activation=tf.nn.relu)

        x = tf.layers.max_pooling2d(x, 2, 2)

        x = tf.layers.conv2d(x, 256, 3, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 512, 3, activation=tf.nn.relu)

        x = tf.layers.max_pooling2d(x, 2, 2)

        x = tf.layers.flatten(x)

        x = tf.layers.dense(x, 2048)

        x = tf.layers.dropout(x, rate=droupout, training=training)

        x = tf.layers.dense(x, 1024)

        x = tf.layers.dropout(x, rate=droupout, training=training)

        out = tf.layers.dense(x, num_classes)

        out = tf.nn.softmax(out) if not training else out


    
    return out




xx = tf.placeholder(tf.float32, shape=[None, 28,28, 1])
yy = tf.placeholder(tf.float32, shape=[None, 10])


train_logits = convnet(xx, 10, training=True, reuse=False)
test_logits = convnet(xx, 10, training=False, reuse=True)


loss_op = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=yy) )

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_op)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:


    sess.run(init_op)

    for i in range(10):


        data, label = mnist.train.next_batch(10)

        _, loss = sess.run([optimizer, loss_op], feed_dict={xx: data.reshape(10, 28, 28, 1), yy: label})

        print(loss)

    
    saver.save(sess, './model.ckpt')