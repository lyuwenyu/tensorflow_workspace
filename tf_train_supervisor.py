import tensorflow as tf 


a = tf.Variable(2.)
b = tf.Variable(3.)
c = tf.add(a, b) 
update = tf.assign(a, c)

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir='./outputs/logs', init_op = init)

global_step = tf.train.get_global_step()

with sv.managed_session() as sess:

    for i in range(100):

        _update = sess.run(update)
        print(_update, i)

        if i % 20 == 0:

            sv.saver.save(sess, './outputs/logs/', global_step=i)

        if i % 30 == 0:

            sv.summary_writer.add_summary(summary='', global_step=i)
            # sv.summary_computed(sess, summary='', global_step=i)

        xx = sess.run(global_step)

        print(xx)
