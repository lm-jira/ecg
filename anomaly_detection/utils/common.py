import csv
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support


def do_roc(scores, true_labels, file_name='', directory='', plot=True):
    """ Does the ROC curve
    Args:
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the ROC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            roc_auc (float): area under the under the ROC curve
            thresholds (list): list of thresholds for the ROC
    """
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)  # compute area under the curve
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.savefig(directory + "/" + file_name + 'roc.png')
        plt.close()

    return roc_auc


def load_data(split="train", length_data=184):
    with open('../data/mitbih_'+split+'.csv', 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')

        if split == "train":
            train_x = []

            for row in rows:
                train_x.append(np.array(row[:length_data]).astype(float))

            train_x = np.expand_dims( np.array(train_x), axis=2)
            return train_x

        elif split == "test":
            test_x = []
            test_y = []

            #i = 0
            for row in rows:
                #if float(row[-1]) == 0:
                #    i+=1
                #if float(row[-1]) == 0 and i >4000:
                #    continue
                test_x.append(np.array(row[:length_data]).astype(float))
                test_y.append((np.array(row[-1]).astype(float)==0).astype(int))
                # label normal=1, anomaly=0

            test_x = np.expand_dims( np.array(test_x), axis=2)
            test_y = np.array(test_y)

            normal_indexes = np.array(test_y==1).astype(int)

            return test_x, test_y


def get_logdir(logname, model):
    if logname == "":
        return "train_log/ECG_"+model
    else:
        return "train_log/ECG_"+ model +"_" + logname

def get_savedir(logname, model):
    if logname == "":
        return "saved/ECG_" + model
    else:
        return "saved/ECG_"+ model + "_" + logname


def get_percentile(scores):
    per = np.percentile(scores, 82)

    return per


def analyse_results(scores, true_labels, logname, model):
    directory = get_savedir(logname, model)
    if not os.path.exists(directory):
        os.makedirs(directory)

    scores = np.array(scores)

    file_name = "ECG_effanogan"
    roc_auc = do_roc(scores, true_labels, file_name=file_name,
                                          directory=directory)

    per = get_percentile(scores)
    y_pred = (scores >= per)
        
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels,
                                                               y_pred.astype(int),
                                                               average='binary')

    print("precision = {}, recall = {}, f1 = {}, auc = {}".format(precision, recall, f1, roc_auc))
    return y_pred


def save_result(input_data, recon_data, logname, number, folder_name, model, prefix="ecg_result"):
    directory = get_savedir(logname, model)
    directory = os.path.join(directory, folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.clf()
    x = []
    y_input = []
    y_recon = []

    for index, (col_input, col_recon) in enumerate(zip(input_data, recon_data)):
        x.append(index*0.008)
        y_input.append(float(col_input))
        y_recon.append(float(col_recon))

    fig = plt.figure(1, figsize=(1,2))

    sub1 = fig.add_subplot(121)
    sub1.plot(x, y_input)
    sub1.title.set_text('input data')
    
    sub2 = fig.add_subplot(122)
    sub2.plot(x, y_recon)
    sub2.title.set_text('reconstructed data')
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.savefig('{}/{}_{}'.format(directory, prefix, number))
