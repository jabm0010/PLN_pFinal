# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:13:57 2019

@author: jabm9
"""

# -*- coding: utf-8 -*-
"""
Author: Salud María Jiménez Zafra
  
Description: Final practice scorer

Last modified: April 9, 2019
"""

import sys

gold_path = 'gold_labels_dev.txt'
input_path = 'resultados.txt'

confusion_matrix = {}
labels = ('positive', 'negative')
for l1 in labels:
    for l2 in labels:
        confusion_matrix[(l1, l2)] = 0

# 1. Read files and get labels
input_labels = {}
with open(input_path, 'r') as input_file:
    for line in input_file.readlines():
        
        try:
            id_file, domain, polarity = line.strip().split('\t')
        except:
            print('Wrong file format: ' + input_path)
            sys.exit(1)
        input_labels[id_file + domain] = polarity
        

with open(gold_path, 'r') as gold_file:
    for line in gold_file.readlines():
        try:
            id_file, domain, true_polarity = line.strip().split('\t')
        except:
            print('Wrong file format: ' + gold_path)
            sys.exit(1)
        
        key = id_file + domain
        if key in input_labels.keys():
            proposed_polarity = input_labels[key]
            confusion_matrix[(proposed_polarity, true_polarity)] += 1
        else:
            print('Wrong file format: ' + input_path)
            sys.exit(1)


### 2. Calculate evaluation measures
avgP = 0.0
avgR = 0.0
avgF1 = 0.0

for label in labels:
    denomP = confusion_matrix[(label, 'positive')] + confusion_matrix[(label, 'negative')]
    precision = confusion_matrix[(label, label)]/denomP if denomP > 0 else 0
    
    denomR = confusion_matrix[('positive', label)] + confusion_matrix[('negative', label)]
    recall = confusion_matrix[(label, label)]/denomR if denomR > 0 else 0
    
    denomF1 = precision + recall
    f1 = 2*precision*recall/denomF1 if denomF1 > 0 else 0
    print('\t' + label + ':\tPrecision=' + "{0:.3f}".format(precision) + '\tRecall=' + "{0:.3f}".format(recall) + '\tF1=' + "{0:.3f}".format(f1) + '\n')
    
    avgP  += precision
    avgR  += recall
    avgF1 += f1

avgP /= 2.0
avgR  /= 2.0
avgF1 /= 2.0

accuracy = (confusion_matrix[('positive','positive')] + confusion_matrix[('negative','negative')]) / (confusion_matrix[('positive','positive')] + confusion_matrix[('negative','negative')] + confusion_matrix[('positive','negative')] + confusion_matrix[('negative','positive')])

print('\nAvg_Precision=' + "{0:.3f}".format(avgP) + '\tAvg_Recall=' + "{0:.3f}".format(avgR) + '\tAvg_F1=' + "{0:.3f}".format(avgF1) + '\tAccuracy=' + "{0:.3f}".format(accuracy))
 
