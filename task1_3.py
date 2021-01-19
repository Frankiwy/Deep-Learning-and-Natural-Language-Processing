import json
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt


def import_data(path):  # function used to import the 2 files
    # 'bert_results_ios.jsonl'
    # 'output_after_Flair_and_BERT_checking_on_dev_set.jsonl'
    results = list()
    with open(path, 'r', encoding='UTF-8') as jsonl_file2:
        file = list(jsonl_file2)

        for e, jsonl_str in enumerate(file):  # go over all the strings in the list
            result = json.loads(jsonl_str)  # load the string as dict
            results.append(result)

        del file
        del result
    return (results)


###############################################################
# (1) Import the data and compute accuracy based on threshold #
###############################################################
bert_results = import_data('lama/bert_results_ios.jsonl')  # the bert results entities
dev_set = import_data('lama/output_after_Flair_and_BERT_checking_on_dev_set.jsonl')  # the dev_set with entites

ps = PorterStemmer()
thr_values = [n for n in np.arange(-0.5, -12, -0.5)]  # thresholds
accuracy_results = list()  # list where we'll store all the accuracies and occurancies based on threshold

for thr in thr_values:
    counter = 0
    for e, d in enumerate(dev_set):

        selected_list = bert_results[e]  # get the list
        selected_list2 = [ps.stem(elm[0].lower()) for elm in selected_list]  # stem all the entites
        entity = ps.stem(d['entity']['mention'].lower())  # steam the entity
        if entity in selected_list2:
            index = selected_list2.index(entity)  # if the entity is in the predicted entites, get its index
            prob = selected_list[index][1]  # use the index to get its prob
            if prob >= thr:  # if the prob is >= of threshold then count it, otherwise no
                counter += 1
    accuracy_results.append([thr, (counter / len(bert_results)) * 100, counter])  # store all the necessary values
    print("Threshold: {}".format(thr), end="\r")

######################################
# (2) Store results into a DataFrame #
######################################

col_names = ['threshold', 'accuracy', 'occurances']
df_thr = pd.DataFrame(accuracy_results, columns=col_names)

##################################
# (3) Plot a Line and a Bar plot #
##################################

fig = plt.figure(figsize=(12, 18))
ax1 = fig.add_subplot(211)  # lineplot
ax2 = fig.add_subplot(212)  # barplot

ax1 = sns.lineplot(x="threshold", y="accuracy", data=df_thr, markers=True, color='red', ax=ax1)
ax2 = sns.barplot(x="threshold", y="occurances", data=df_thr, palette="GnBu_d", ax=ax2)

for ax in [ax1, ax2]:
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

ax1.set_title('LinePlot', color='orange', fontsize=30, pad=15)
ax2.set_title('BarPlot', color='orange', fontsize=30, pad=15)

plt.show();