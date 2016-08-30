
# Script to predict survivability of passengers on Titanic
# Re-implementation of k-Nearest Neighbor example in Data Science from Scratch
# https://github.com/joelgrus/data-science-from-scratch
# linear_algebra module required. See the link above.


import random, csv
from collections import Counter
from linear_algebra import distance, get_column

fileI = 'titanic3.csv'
pclass, survived, sex, age, fare = [], [], [], [], []           # Sets variables
matrix, res = [], []                                                     # Lists created

def file_in(fileI):
    with open(fileI, 'r', encoding='utf8',newline='') as infile:    # Opens file as read
        reader = csv.reader(infile, delimiter=',')                  # Defines csv conditions
        next(reader, None)                                          # Skips heading
        for row in reader:                                          # Iterates rows
            pclass = int(row[0])                                    # Creates var for the attributes in assoc columns
            survived = int(row[1])
            try:                                                    # Exceptions for skipping null entries
                sex = str(row[3])
            except:
                continue
            try:                                            # Try and except are used to skip lines with bad data
                age = float(row[4])
            except:
                continue
            try:
                fare = float(row[8])
            except:
                continue                                    # Inserts zero for missing values

            dat = [pclass, sex, age, fare]                  # Stores for appending to the attribute matrix
            matrix.append(dat)                              # Matrix of attributes
            res.append(survived)                            # Separate vector for results

file_in(fileI)                                              # Calls function to import data

random.seed(1)                                              # Used to change the spread of the data split

x = matrix                                                  # Matrix of attributes
y = res                                                     # Vectors of results

def split_data(data, prob):                                         # Splits data into test and train
    results = [], []                                                # based on random probability
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)                    # Split the dataset of pairs
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return [x_train, x_test, y_train, y_test]                       # Returns the matrix and vectors of the train and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, 0.33)     # Outputs the matrix and the vectors used for model

def majority_vote(labels):
    vote_counts = Counter(labels)                                   # Function reduces the distance of nearest points
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner                                               # When the number of winners is 1, it is returned
    else:
        return majority_vote(labels[:-1])

def knn_classify(k, labeled_points, train_point):
    by_distance = sorted(labeled_points, key=lambda x: distance(x[0], train_point))    # order the labeled points from nearest to farthest
    k_nearest_labels = [label for _, label in by_distance[:k]]
    return majority_vote(k_nearest_labels)

def ranges(dat):
    d = int(min(dat)), int(max(dat))
    return d

def classify(num, t1, t2):              # Arguments are: number of nearest neigbors, attribute 1, attribute 2

    tst1_d = get_column(x_test, t1)
    tst2_d = get_column(x_test, t2)
    surv_d = y_train
    all_list, results = [], []

    att1_d = get_column(x_train, t1)                                    # Column attributes for the training set
    att2_d = get_column(x_train, t2)

    for x,y,z in zip(surv_d, att1_d, att2_d):                           # Dictionary for the training data
        dat = ([y,z], str(x))                                           # Converting result to a string
        all_list.append(dat)

    for att1, att2 in zip(tst1_d, tst2_d):                              # Iterates over the training set
            predict_surv = knn_classify(num, all_list, [att1, att2])    # The test data set is run on the training set
            results.append(predict_surv)

    return results

def confusion(results, y_test):

    def precision(tp, fp):                                      # Precision of results is calculated
        return tp / (tp + fp)

    def recall(tp, fn):                                         # Recall of results is calculated
        return tp / (tp + fn)

    conf_dict = {'TN': 0, 'TP': 0, 'FP': 0, 'FN': 0}            # True negative, True positive, False positive, False negative

    for i, j in zip(results, y_test):                           # Conditional loop to determine accuracy of prediction
        i = int(i)
        if i == 0 and j == 0:
            conf_dict['TN'] += 1
        elif i == 1 and j == 1:
            conf_dict['TP'] += 1
        elif i == 0 and j == 1:
            conf_dict['FP'] += 1
        elif i == 1 and j == 0:
            conf_dict['FN'] += 1

    prec = precision(conf_dict['TP'], conf_dict['FP'])          # Calls and stores results
    rec = recall(conf_dict['TP'], conf_dict['FN'])

    return conf_dict, prec, rec


run_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]       # Runs the attributes in sequence

for l in x_train, x_test:                                   # Converts males to 0 and females to 1
    for s in l:
        if s[1] == 'male':
            s[1] = 0
        else:
            s[1] = 1

for r in run_list:                                          # Prints prediction results and
    results = classify(3, r[0], r[1])                       # confusion matrix with TP, TN, FP, FN
    print('Results for %s and %s' %(r[0], r[1]))
    print(confusion(results, y_test))











