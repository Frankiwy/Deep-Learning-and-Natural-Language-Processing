# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
import lama.options as options
import json
import jsonlines
import numpy as np
import pprint as pp
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def main(args,path_no_masked, path_masked, class_label):

    vectors = list()

    # load the file with entities
    dev_set= list()
    with open(path_no_masked, 'r', encoding='UTF-8') as jsonl_file2:
        file = list(jsonl_file2)

        for e,jsonl_str in enumerate(file): #go over all the strings in the list
            result = json.loads(jsonl_str) # load the string as dict
            dev_set.append(result)

        del file
        del result

    masked_dev_set=list()
    with open(path_masked, "r", encoding="UTF-8") as jsonl_file:
        masked_file = list(jsonl_file)

        for e,jsonl_str in enumerate(masked_file): # go over all the strings in the list

            get_dict = json.loads(jsonl_str) # load the string as dict
            masked_dev_set.append(get_dict)

        del masked_file
        del get_dict


        print("Language Models: {}".format(args.models_names))
        models = {}
        for lm in args.models_names:
            models[lm] = build_model_by_name(lm, args)
        for model_name, model in models.items():
            print("\n{}:".format(model_name))
            if args.cuda:
                model.try_cuda()

            for e, d in enumerate(masked_dev_set):
                get_key = str(dev_set[e]['id']) #get id
                not_masked_sentence = [dev_set[e]['claim']] #get no masked sentence
                masked_sentence = [d[get_key]] # get masked sentence



                sentences = [masked_sentence, not_masked_sentence]

                contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
                    sentences)

                find_index = tokenized_text_list[0].index("[MASK]") #get index


                vector_masked = contextual_embeddings[11][0][find_index].numpy()
                vector_no_masked = contextual_embeddings[11][1][find_index].numpy()
                concatenated_vector = np.concatenate((vector_masked, vector_no_masked))

                if class_label == 'yes':

                    claim_label = dev_set[e]["label"]  # see if SUPPORTED or REFUSED

                    vectors.append({
                        'vector': concatenated_vector.tolist(),
                        'label': claim_label})
                else:
                    vectors.append({
                        'vector': concatenated_vector.tolist()})

    return vectors

if __name__ == '__main__':

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    # PART 1: get the contextual representation #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

    parser = options.get_general_parser()
    parser.add_argument('--cuda', action='store_true', help='Try to run on GPU')
    args = options.parse_args(parser)
    returned_dev_set_vectors = main(args,
                            'lama/output_after_Flair_and_BERT_checking_on_dev_set.jsonl',
                            'lama/masked_claim_with_ID_on_dev_set.jsonl',
                                    class_label='yes'
                                    )
    print('Done with dev_set')
    print()
    returned_NLP_test_vectors = main(args,
                            'lama/singletoken_test_fever_homework_NLP.jsonl',
                            'lama/masked_claim_singletoken_test_fever_homework_NLP.jsonl',
                                     class_label='no')
    print('Done with NLP_test')


    with jsonlines.open('vectors_dev_set.jsonl',mode='w') as writer:
        writer.write_all(returned_dev_set_vectors)


    with jsonlines.open('vectors_NLP_test.jsonl',mode='w') as writer:
        writer.write_all(returned_NLP_test_vectors)

    del returned_NLP_test_vectors
    del returned_dev_set_vectors



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    # PART 2: Perform classification and prediction #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

    print('Let\'s start classification!')

    def get_df(path, label):  # function used to takes in input the vectors and returns a dataframe
        # 'vectors_dev_set.jsonl'

        vector_list = list()

        with open(path, 'r', encoding='UTF-8') as jsonl_file:
            file = list(jsonl_file)

            for e, jsonl_str in enumerate(file):  # go over all the strings in the list
                result = json.loads(jsonl_str)  # load the string as dict
                vector = np.array(result['vector'])  # from list to vector

                if label == 'yes':
                    vector_and_label = np.append(result['label'], vector)  # add label to vector
                    vector_list.append(vector_and_label)
                else:
                    vector_list.append(vector)
        df = pd.DataFrame(list(map(np.ravel, vector_list)))  # convert the list into a DataFrame

        if label == 'yes':
            columns_names = ['label'] + ['v ' + str(n) for n in range(1, 1537)]  # columns names
            df.columns = columns_names  # change columns' names
        else:
            columns_names = ['v ' + str(n) for n in range(1, 1537)]  # columns names
            df.columns = columns_names  # change columns' names
        return (df)


    df = get_df('lama/vectors_dev_set.jsonl', label='yes')

    ###############
    # (a) dev set #
    ###############

    # (a.1) perform standardization and PCA on dev_set

    pca = PCA(n_components=250)  # defines the PCA object
    x = StandardScaler().fit_transform(df.iloc[:, 1:])  # standardize the data
    pca_transformation = pca.fit_transform(x)  # perfom PCA

    columns_names = ['PCA ' + str(n) for n in range(1, 251)]
    df_pca = pd.DataFrame(pca_transformation, columns=columns_names)  # return data into df
    df_pca['label'] = df['label']

    # (a.2) prepare data to be splitted into train and test set
    X = df_pca.drop(['label'], axis='columns')  # drop the label column from our df
    y = df_pca.label  # create a serie where store all the labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)  # 30%test

    # (a.3) A classifier is defined. Where there are the selected models
    classifiers = {
            'knn': KNeighborsClassifier(),
            'dtc': DecisionTreeClassifier(),
            'svm': SVC(),
            'rf': RandomForestClassifier(),
            'ada': AdaBoostClassifier()
    }

    # (a.4) parameter dict where there are all the parameters that have to be tuned.
    paramiters = {
            'knn': {'n_neighbors': [int(x) for x in np.linspace(1, 200, num=5)], 'weights': ['uniform', 'distance']},

            'dtc': {'max_depth': [5, 10], 'max_features': [int(x) for x in np.linspace(1, 23, num=5)]},

        'svm': [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],

            'rf': {
                'n_estimators': [int(x) for x in np.linspace(10, 500, num=3)],
                'max_depth': [int(x) for x in np.linspace(10, 100, 3)],
                'bootstrap': [True, False]

            },
            'ada': {
                'n_estimators': [500, 1000, 2000],
                'learning_rate': [0.01, .1, 1]
            }
    }

    # (a.5) perform GridSearch for all models

    gs_scores = {}  # dictionary where will be stored all the scores
    gs_paramiters = {}  # dictionary where will be stored all the tested paramiters
    gs_estimators = {}  # dictionary where will be stored all the estimators
    for k in classifiers.keys():  # we run up to all the models are tuned
        grid = GridSearchCV(
            classifiers[k],
            paramiters[k],
            cv=KFold(n_splits=10, random_state=25, shuffle=True),
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        gs_scores[k] = grid.cv_results_
        gs_paramiters[k] = grid.best_params_
        gs_estimators[k] = grid.best_estimator_

    accuracy_results = defaultdict()  # dictionary where we are going to store the accuracy for each model
    confusion_matrices = defaultdict()  # the confusion matrix
    cross_validation = defaultdict()  # cross validation results

    for k in gs_estimators.keys():  # now the models are called one-by-one using the best configurations

        model_selected = gs_estimators[k]  # select the model for the tuned models
        cross_validation_score = cross_val_score(model_selected, X_train, y_train, cv=10)
        model_selected.fit(X_train, y_train)  # fit the model with the train set
        y_prediction = model_selected.predict(X_test)  # make the prediction

        cross_validation[k] = cross_validation_score.tolist()  # store cross validation scores
        accuracy_results[k] = accuracy_score(y_test, y_prediction)  # store accuracy
        confusion_matrices[k] = confusion_matrix(y_test, y_prediction)  # store confusion matrices

    print('Best accuracy per each model')
    pp.pprint(accuracy_results)


    ###############
    # (b) NLP set #
    ###############

    df_NLP = get_df('lama/vectors_NLP_test.jsonl', label='no')  # get the dataframe

    # (b.1) performing standardization and PCA on NLP datatset

    pca = PCA(n_components=250)  # defines the PCA object
    x = StandardScaler().fit_transform(df_NLP)  # standardize the data
    pca_transformation = pca.fit_transform(x)  # perfom PCA

    columns_names = ['PCA ' + str(n) for n in range(1, 251)]
    df_NLP_pca = pd.DataFrame(pca_transformation, columns=columns_names)  # return data into df

    # (b.2) the best model is used to perform the prediction on the NLP dataset
    print('Performs prediction using the best model')
    predictions = gs_estimators['svm'].predict(df_NLP_pca)

    with jsonlines.open('predictions.jsonl', mode='w') as writer:  # write intermediate data into file
        writer.write_all(predictions)
    print('Prediction stored into predictions.jsonl file')







