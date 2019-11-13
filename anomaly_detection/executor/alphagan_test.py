import tensorflow as tf
import importlib
import numpy as np
import os

from utils import common


def _test(logname):
    logdir = common.get_logdir(logname, "alphagan")
    network = importlib.import_module("network.alphagan")

    # Parameter
    batch_size = network.batch_size
    latent_dim = network.latent_dim

    # Data features
    len_data = 184

    # Output node name
    scores_l1_str = "Testing/Scores/score_l1:0"
    scores_l2_str = "Testing/Scores/score_l2:0"
    #recon_img_str = 'generator_model/generator/ExpandDims:0'
    recon_img_str = 'generator_model/generator/sigmoid:0'
    fetches_str = [scores_l1_str, scores_l2_str, recon_img_str]

    test_x, test_label = common.load_data(split="test", length_data=len_data)
    nr_batches_test = len(test_label)//batch_size
    leftover_flg = len(test_label) % batch_size != 0
    if leftover_flg:
        nr_batches_test += 1

    scores = []
    recon_data = []

    with tf.Session() as sess:
        states = tf.train.get_checkpoint_state(logdir)
        chk_paths = states.all_model_checkpoint_paths
        latest_path = chk_paths[-1]
        saver = tf.train.import_meta_graph(latest_path + ".meta")
        saver.restore(sess, latest_path)

        all_vars = tf.get_default_graph().as_graph_def().node
        #for var in all_vars:
        #    if "generator_model/generator/" in var.name:
        #        print(var.name)

        for t in range(nr_batches_test):
            if leftover_flg and t == nr_batches_test-1:
                # the last batch and the batch is not full
                batch_x = test_x[t*batch_size:]
                batch_label = test_label[t*batch_size:]

                size = len(batch_x)
                new_shape = [batch_size - size] + list(batch_x[0].shape)
                fill = np.ones(new_shape)
                batch_data = np.concatenate([batch_x, fill], axis=0)
            else:
                batch_data = test_x[t*batch_size:(t+1)*batch_size]
                batch_label = test_label[t*batch_size:(t+1)*batch_size]

                size = batch_size

            feed_dict = {"input_x:0": batch_data,
                         "input_z:0": np.random.normal(size=[batch_size, latent_dim]),
                         "is_training_pl:0": False}

            l1score, l2score, reconst_data = sess.run(fetches_str, feed_dict=feed_dict)
            scores.append(l1score[:size])
            recon_data.append(reconst_data)

        scores = np.concatenate(scores, axis=0)
        recon_data = np.concatenate(recon_data, axis=0)

    y_pred = common.analyse_results(scores, test_label, logname,"alphagan")
    fn_index = np.logical_and(np.logical_not(y_pred), (np.array(test_label)==1))

    i = 0
    for index, val in enumerate(fn_index):
        if val:
            i+=1                
            common.save_result(test_x[index], recon_data[index], logname, index, folder_name="false_negative_samples", model="alphagan")
        if i>20:
            break

    tp_index = np.logical_and(y_pred, (np.array(test_label)==1))
    i = 0
    for index, val in enumerate(tp_index):
        if val:
            i+=1                
            common.save_result(test_x[index], recon_data[index], logname, index, folder_name="true_positive_samples", model="alphagan")
        if i>20:
            break
    
    #print("false negative = ", np.sum(fn_index.astype(int)))
    #print("score   =  ",scores)


def run(logname, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _test(logname)
