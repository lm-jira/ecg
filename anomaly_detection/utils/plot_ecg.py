import matplotlib.pyplot as plt
import csv
import numpy as np 


full_label_class = {0:"Normal", 1:"Supraventricular ectopic beat", 2:"Ventricular ectopic beat", 3:"Fusion of N and V", 4:"Paced beat"}
short_label_class = {0:"Normal", 1:"S", 2:"V", 3:"F", 4:"Q"}


def plot_hist(split="train"):
    with open('data/mitbih_'+split+'.csv', 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')

        labels = []
        for row in rows:
            labels.append(float(row[-1]))

        unique_label = set(labels)
        num_label = []
        for l in unique_label:
            num_label.append(labels.count(l))
        
        plt.bar([short_label_class[l] for l in list(unique_label)], num_label)
        plt.show()


def plot_data(split="train"):
    with open('data/mitbih_'+split+'.csv', 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
    
        temp=0
        i=0
        for j,row in enumerate(rows):
            x = []
            y = []
        
            if float(row[-1]) < temp:
                continue
            print(j,"  label = ", row[-1])
            i = i+1
            for index, col in enumerate(row[:-1]):    
                x.append(index*0.008)
                y.append(float(col))
            plt.plot(x, y)
            plt.show()
            temp = temp+1
            if i > 10:
                exit()

def plot_ecg_sample(sample):
    for j,row in enumerate(sample):
        plt.clf()
        if j == 10:
            return
        x = []
        y = []
        
        for index, col in enumerate(row[:-1]):    
            x.append(index*0.008)
            y.append(float(col))
        plt.plot(x, y)
        plt.show()
        plt.savefig('ecg_{}'.format(j))

        
plot_hist()
