
import glob
import tensorflow as tf

imgs_path = glob.glob('../../dataset/*.jpg')



### dataset.py 

with open('./dataset.txt', 'w') as f:
    for ii, ll in enumerate(imgs_path):

        f.write( '{}\t{}\n'.format(ll, ii))


def get_dataset(path, is_training=True):


    def _process(items):

        path = items['image_path']
        label = items['image_label']

        img = tf.read_file(path)
        # img = tf.gfile.FastGFile(path, 'rb').read()

        img = tf.image.decode_jpeg(img, channels=3)
    
        img = tf.image.resize_images(img, [224,224])

        img = img * 2.0/255 + 1.
        img = tf.reshape(img, [224,224] + [3])

        return img, label


    with open(path, 'r') as f:

        lines = f.readlines()

    paths = [ ll.strip().split('\t')[0] for ll in lines]
    labels = [ int(ll.strip().split('\t')[1]) for ll in lines]

    dataset = tf.data.Dataset.from_tensor_slices( {'image_path': paths, 'image_label': labels} )
    dataset = dataset.shuffle(100)
    dataset = dataset.map(_process).repeat(5).batch(3)

    iterator = dataset.make_one_shot_iterator()
    
    img_iter, lab_iter = iterator.get_next()
    
    return img_iter, lab_iter





### model.py


def convnet(inputs, num_classes, is_training, reuse):

    with tf.variable_scope('convnet'): #, reuse=reuse):


        conv1 = tf.layers.conv2d(inputs, filters=10, kernel_size=[7,7], strides=2, activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.average_pooling2d(conv1, pool_size=[7,7], strides=7)
        pool1 = tf.layers.batch_normalization(pool1, training=is_training)

        conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=[3,3], strides=2, activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.average_pooling2d(conv2, pool_size=[7,7], strides=1)
        
        
        #pool2 = tf.reshape(pool2, [-1, 128])
        pool2 = tf.layers.flatten(pool2)

        fc1 = tf.layers.dense(pool2, 1024)

        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        fc2 = tf.layers.dense(fc1, num_classes)

        logits = tf.nn.softmax(fc2) if not is_training else fc2
        ###
    return logits

    

def build_model(inputs, targets, is_training=True, num_classes=6):


    print('build model ... ')

    logits = convnet(inputs, num_classes, is_training=is_training, reuse=not is_training)


    step = tf.train.get_or_create_global_step()

    ops_dict = {}

    accuracy = tf.reduce_mean( tf.cast( tf.equal(tf.cast(targets, tf.int64), tf.argmax(logits, axis=-1)) , tf.float32))

    # acc_top = tf.reduce_mean( tf.cast( tf.nn.in_top_k(logits, targets, k=3), tf.float32))
    ops_dict['accuracy'] = accuracy


    prefix = 'train' if is_training else 'test'

    tf.summary.scalar('{}/acc'.format(prefix), accuracy)

    if is_training:

        targets = tf.one_hot(indices=targets, depth=num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=logits)

        # loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        learning_rate = tf.train.exponential_decay(0.1, global_step=step, decay_steps=5, decay_rate=0.1, staircase=True)

        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(update_op)

        with tf.control_dependencies(update_op):
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=step)

        tf.summary.scalar('{}/loss'.format(prefix), loss)
        tf.summary.scalar('{}/lr'.format(prefix), learning_rate)

        ops_dict['loss'] = loss
        ops_dict['train_op'] = train_op

        ops_dict['lr'] = learning_rate


    tf.summary.image(prefix+'/input', inputs)
    summary_op = tf.summary.merge_all()


    ops_dict['summary_op'] = summary_op
    ops_dict['step'] = step
    ops_dict['params'] = tf.trainable_variables()
    ops_dict['logits'] = logits


    print('build model done...')
    return ops_dict


### train.py

def fineturn(variables, ckpt_path='./logs'):


    print('----fineturn-----')
    varss = tf.train.list_variables(ckpt_path)
    www = tf.train.load_variable(ckpt_path, 'batch_normalization/beta')
    
    ops = []

    for v in variables:

        if v.name == 'global_step':
            continue 
        
        for vv in varss:

            if vv[0] in v.name : #and vv[1] == v.shape.as_list(): ###

                    ops += [ tf.assign( v, tf.train.load_variable(ckpt_path, vv[0]) ) ]

    if len(ops):
        return tf.group(*ops)







def train(path='./dataset.txt', e=0):
    

    with tf.Graph().as_default() as graph:
        

        img_iter, lab_iter = get_dataset(path)
        # test_img_iter, test_lab_iter = get_dataset(path)


        ops = build_model(img_iter, lab_iter, is_training=True)
        # test_ops = build_model(test_img_iter, test_lab_iter, is_training=False)


        # params_names = [ v.name for v in ops['params']]

        # for ii, pp in enumerate(ops['params']):
        #     print(ii, pp.get_shape(), pp.name)

        # for ii, pp in enumerate(tf.all_variables()):
        #     print(ii, pp.name, pp.get_shape())

        # tf.summary.scalar('train/loss', ops['loss'])
        # tf.summary.scalar('train/acc', ops['accuracy'])
        # summ_op = tf.summary.merge_all()


        # with tf.Session() as sess:

        #     sess.run(tf.global_variables_initializer())

        #     while True:
                
        #         try:
                    
        #             # x, y = sess.run([img_iter, lab_iter])
        #             # out = sess.run(convnet)
        #             cost = sess.run(ops['loss'])

        #             sess.run(ops['train_op'])

        #             # print(x.shape,  y)
        #             # print(out.shape)


        #         except tf.errors.OutOfRangeError:

        #             print('done..')
        #             break

        # step = tf.train.get_or_create_global_step()


        # fineturn_op = fineturn(tf.global_variables())

        sv = tf.train.Supervisor(graph=graph, logdir='./logs', save_model_secs=3)

        with sv.managed_session() as sess:


            if sess.run(ops['step']) < 10:
                # sess.run(fineturn_op)
                pass



            while True:  ## train

                import time
                time.sleep(1)

                try:

                    _, cost, summ = sess.run([ops['train_op'], ops['loss'], ops['summary_op']])

                    print( sess.run(ops['step']) )
                    print( sess.run(ops['lr']) )
                    print( cost )
                    print( sess.run(lab_iter))
                    print('---')
                    
                    sv.summary_writer.add_summary(summ, sess.run(ops['step']) )


                except tf.errors.OutOfRangeError:

                    print('done...{}'.format(e))
                    
                    # sv.saver.save(sess, './logs/model/ckpt')

                    break



                


def test(path='./dataset.txt'):
    

    with tf.Graph().as_default() as graph:


        img_iter, lab_iter = get_dataset(path)

        ops = build_model(img_iter, lab_iter, num_classes=6, is_training=False)
        step = tf.train.get_or_create_global_step()


        # tf.summary.scalar('test/acc', ops['accuracy'])
        # summ_op = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables())
        # saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('./logs')

        with tf.Session(graph=graph) as sess:

            saver.restore(sess, tf.train.latest_checkpoint('./logs'))

            summ, acc = sess.run([ops['summary_op'], ops['accuracy']])
            summary_writer.add_summary(summ)

            print(sess.run(step), sess.run(ops['step']), acc)


    print('test done...')

### inference.py

def inference():
    pass



def main(_):

    for e in range(10):
       train(e=e)
       test()
    
    
if __name__ == '__main__':

    tf.app.run()

