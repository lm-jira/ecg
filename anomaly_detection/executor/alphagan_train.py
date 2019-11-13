import tensorflow as tf
import importlib
import numpy as np
import os

from utils import common
# avoid log(0)
EPS = 1e-12

# Parameters
reconstruction_loss_weight = 40.0


def _train(nb_epochs, weight, logname):
    logdir = common.get_logdir(logname, "alphagan")
    network = importlib.import_module("network.alphagan")

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim

    # Data features
    len_data = 184

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Placeholder
    x_pl = tf.placeholder(tf.float32, shape=[None, len_data, 1], name="input_x")
    z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim], name="input_z")
    is_training_pl = tf.placeholder(tf.bool, [], name="is_training_pl")
    learning_rate_pl = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Models
    enc = network.encoder
    gen = network.generator
    dis = network.discriminator
    code_dis = network.code_discriminator

    with tf.variable_scope("encoder_model"):
        z_gen = enc(x_pl, is_training=is_training_pl)

    with tf.variable_scope("generator_model"):
        rec_x = gen(z_gen, is_training=is_training_pl)

    with tf.variable_scope("generator_model"):
        x_gen = gen(z_pl, is_training=is_training_pl, reuse=True)

    x_fake = tf.concat([rec_x, x_gen], 0)

    with tf.variable_scope("discriminator_model"):
        y_fake = dis(x_fake, is_training=is_training_pl)

    with tf.variable_scope("discriminator_model"):
        y_real = dis(x_pl, is_training=is_training_pl, reuse=True)

    with tf.variable_scope("code_discriminator_model"):
        c_fake = code_dis(z_gen)

    with tf.variable_scope("code_discriminator_model"):
        c_real = code_dis(z_pl, reuse=True)

    # Loss
    with tf.name_scope("loss_function"):
        reconstruction_loss = reconstruction_loss_weight * tf.reduce_mean(tf.abs(x_pl - rec_x))
        generator_loss = tf.reduce_mean(-tf.log(y_fake + EPS))
        discriminator_loss = -tf.reduce_mean(tf.log(y_real + EPS)) - tf.reduce_mean(tf.log(1 - y_fake + EPS))
        code_discriminator_loss = -tf.reduce_mean(tf.log(c_real + EPS)) - tf.reduce_mean(tf.log(1 - c_fake + EPS))
        code_generator_loss = tf.reduce_mean(-tf.log(c_fake + EPS))

        # loss for autoencoder
        encoder_generator_loss = reconstruction_loss + generator_loss + code_generator_loss

    # Optimizer
    with tf.name_scope('optimizers'):
        tvars = tf.trainable_variables()
        dvars = [
            var for var in tvars if 'discriminator_model' in var.name]
        cvars = [
            var for var in tvars if 'code_discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        beta1 = 0.5
        beta2 = 0.9

        # Auto Encoder
        encoder_generator_opt = tf.train.AdamOptimizer(learning_rate_pl, beta1, beta2).minimize(
            encoder_generator_loss,
            var_list=evars + gvars)

        # ------------------------------------------------
        # Discriminatorr
        discriminator_opt = tf.train.AdamOptimizer(learning_rate_pl, beta1, beta2).minimize(
            discriminator_loss,
            var_list=dvars)

        # ------------------------------------------------
        # Code Discriminatorr
        code_discriminator_opt = tf.train.AdamOptimizer(learning_rate_pl, beta1, beta2).minimize(
            code_discriminator_loss,
            var_list=cvars)

    with tf.name_scope('training_summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('discirminator_loss', discriminator_loss, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('generator_loss', generator_loss, ['gen'])

        with tf.name_scope('enc_gen_summary'):
            tf.summary.scalar('encoder_generator_loss', encoder_generator_loss, ['enc_gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_enc_gen = tf.summary.merge_all('enc_gen')
        sum_op = tf.summary.merge([sum_op_dis, sum_op_gen, sum_op_enc_gen])

    # Testing
    with tf.name_scope("Testing"):
        with tf.variable_scope('Scores'):
            rec = x_pl - rec_x
            rec = tf.contrib.layers.flatten(rec)
            score_l1 = tf.norm(rec, ord=1, axis=1,
                               keep_dims=False, name='d_loss')
            score_l1 = tf.squeeze(score_l1, name="score_l1")

            rec = x_pl - rec_x
            rec = tf.contrib.layers.flatten(rec)
            score_l2 = tf.norm(rec, ord=2, axis=1,
                               keep_dims=False, name='d_loss')
            score_l2 = tf.squeeze(score_l2, name="score_l2")
                                                                                                    
    sv = tf.train.Supervisor(logdir=logdir)
    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    with sv.managed_session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0

        train_x = common.load_data(split="train", length_data=len_data)
        nr_batches = len(train_x)//batch_size
        
        while epoch < nb_epochs:
            print("epoch{} ".format(epoch))
            #lr_decay = (epoch//(nb_epochs/3))*0.75
            lr_decay = 1
            lr = starting_lr * lr_decay

            shuffle_indexes = np.random.permutation(len(train_x))
            train_x = train_x[shuffle_indexes]

            train_loss_dis, train_loss_gen_enc, train_loss_code_dis = [0, 0, 0]

            for t in range(nr_batches):
                batch_x = train_x[t*batch_size:(t+1)*batch_size]

                feed_dict = {x_pl: batch_x,
                             z_pl: np.random.normal(size=[batch_size, latent_dim]),
                             is_training_pl: True,
                             learning_rate_pl: lr}
                # ---------------------------------------------------
                _, eg_loss = sess.run([encoder_generator_opt, encoder_generator_loss], feed_dict=feed_dict)
                _, eg_loss = sess.run([encoder_generator_opt, encoder_generator_loss], feed_dict=feed_dict)
                # ---------------------------------------------------
                _, d_loss = sess.run([discriminator_opt, discriminator_loss], feed_dict=feed_dict)
                # ---------------------------------------------------_, _, _, ld, ldxz, ldxx, ldzz, step = sess.run([train_dis_op_xz,
                _, c_loss = sess.run([code_discriminator_opt, code_discriminator_loss], feed_dict=feed_dict)

                train_loss_dis += d_loss
                train_loss_gen_enc += eg_loss
                train_loss_code_dis += c_loss

                print("dis loss = ", train_loss_dis)
                print("gen enc loss = ", train_loss_gen_enc)
                if t % 100 == 0:
                    sm = sess.run(sum_op, feed_dict=feed_dict)
                    writer.add_summary(sm, epoch*nr_batches+t)

            train_loss_gen_enc /= nr_batches
            train_loss_code_dis /= nr_batches
            train_loss_dis /= nr_batches

            epoch += 1

        sv.saver.save(sess, logdir+"/model.ckpt")


def run(nb_epochs, weight, logname, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _train(nb_epochs, weight, logname)
