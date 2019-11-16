import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs

def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""
def cross_validate(K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    length = len(X)
    batchlength = 1.0*length/K
    indices = np.array([i for i in range(len(X))])
    shuffle_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]    
    for k in range(K):
        testindices = np.array([i for i in range(int(k*batchlength),int((k+1)*batchlength))])
        trainindices = np.array([i for i in indices if i not in testindices])

        xtest = X[testindices]
        ytest = Y[testindices]
        xtrain = X[trainindices]
        ytrain = Y[trainindices]
        
        for i, params in enumerate(parameters):
            print(k, i , params)
            learner = Algorithm(params)
            learned_parameters = learner.learn(xtrain, ytrain)
            predictions = learner.predict(xtest, learned_parameters)
            error = geterror(ytest,predictions)
            all_errors[i][k] = error

    avg_errors = np.mean(all_errors, axis=1)
    print(all_errors)
    print(avg_errors)
    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])

    best_parameters = parameters[np.argmin(avg_errors)]
    print(best_parameters)
    return best_parameters

def stratified_cross_validate(K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    batchlength = 1.0*len(X)/K
    indices = np.array([i for i in range(len(X))])
    shuffle_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(shuffle_indices)
    
    # Split into two indices, the ones with y=0 and with y=1
    # Take len(group1)/K, len(group2)/K
    # create add these together, and shuffle the data and add to an empty array
    
    group0_indices = np.array([])
    group1_indices = np.array([])
    for i in shuffle_indices:
        if Y[i] == 0:
            group0_indices = np.concatenate((group0_indices, [i]))
        if Y[i] == 1:
            group1_indices = np.concatenate((group1_indices, [i]))

    group0_indices = group0_indices.astype(int)
    group1_indices = group1_indices.astype(int)    
    
    group0_batchlength = 1.0*len(group0_indices)/K
    group1_batchlength = 1.0*len(group1_indices)/K  
    print(group0_batchlength)
    print(group1_batchlength)  
    final_shuffled_indices = np.array([])

    
    for k in range(K):
        temp_indices = np.array([])

        temp_indices = np.concatenate((temp_indices, group0_indices[int(k*group0_batchlength):int((k+1)*group0_batchlength)]))
        temp_indices = np.concatenate((temp_indices, group1_indices[int(k*group1_batchlength):int((k+1)*group1_batchlength)]))

        np.random.shuffle(temp_indices)
        print(len(temp_indices))
        final_shuffled_indices = np.concatenate((final_shuffled_indices, temp_indices))
    
    final_shuffled_indices = final_shuffled_indices.astype(int)
    
    X = X[final_shuffled_indices]
    Y = Y[final_shuffled_indices]

    for k in range(K):
        testindices = np.array([i for i in range(int(k*batchlength),int((k+1)*batchlength))])
        trainindices = np.array([i for i in indices if i not in testindices])

        xtest = X[testindices]
        ytest = Y[testindices]
        xtrain = X[trainindices]
        ytrain = Y[trainindices]
        
        for i, params in enumerate(parameters):
            print(k, i , params)
            learner = Algorithm(params)
            learned_parameters = learner.learn(xtrain, ytrain)
            predictions = learner.predict(xtest, learned_parameters)
            error = geterror(ytest,predictions)
            all_errors[i][k] = error
       
    avg_errors = np.mean(all_errors, axis=1)
    print(all_errors)
    print(avg_errors)
    for i, params in enumerate(parameters):
        print('Stratified cross validate parameters:', params)
        print('average error:', avg_errors[i])

    best_parameters = parameters[np.argmin(avg_errors)]
    print(best_parameters)
    return best_parameters
    
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=10,
                        help='Specify the number of runs')
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset



    classalgs = {
        'Random': algs.Classifier,
      #  'Naive Bayes': algs.NaiveBayes, #done
      #  'Linear Regression': algs.LinearRegressionClass, #done
      #  'Logistic Regression': algs.LogisticReg,
      #  'Neural Network 1 hidden layer': algs.NeuralNet_1hiddenlayer,
      #   'Neural Network 2 hidden layers': algs.NeuralNet_2hiddenlayers,
      #  'Linear Kernel Logistic Regression': algs.LinearKernelLogisticRegression,
      #  'Hamming Distance Kernel Logistic Regression': algs.HammingDistanceKernelLogisticRegression,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Linear Regression': [{'regwgt': 0.01}],
        'Naive Bayes': [
            # first set of parameters to try
            #{ 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Logistic Regression': [
            { 'stepsize': 0.001 },
            { 'stepsize': 0.01 },
        ],
        'Neural Network 1 hidden layer': [
            { 'epochs': 100, 'nh': 4 },
            { 'epochs': 100, 'nh': 8 },
        #    { 'epochs': 100, 'nh': 16 },
        #    { 'epochs': 100, 'nh': 32 },
        ],
        'Neural Network 2 hidden layers': [
        #    { 'epochs': 100, 'nh1': 4, 'nh2': 4 },
            { 'epochs': 100, 'nh1': 8, 'nh2': 8 },
        #    { 'epochs': 100, 'nh1': 16, 'nh2': 16 },
        #    { 'epochs': 100, 'nh1': 32, 'nh2': 32 },
        ],
        'Linear Kernel Logistic Regression': [
        #    { 'centers': 10, 'stepsize': 0.01 },
        #    { 'centers': 20, 'stepsize': 0.01 },
            { 'centers': 40, 'stepsize': 0.01 },
        #    { 'centers': 80, 'stepsize': 0.01 },
        ],
        'Hamming Distance Kernel Logistic Regression': [
            { 'centers': 10, 'stepsize': 0.01 },
        #    { 'centers': 20, 'stepsize': 0.01 },
        #    { 'centers': 40, 'stepsize': 0.01 },
        #    { 'centers': 80, 'stepsize': 0.01 },
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    errors_scv = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)
        errors_scv[learnername] = np.zeros(numruns)
        
    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        

        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        best_parameters = {}
        best_parameters_scv = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])
            best_parameters[learnername] = params[0]
            best_parameters_scv[learnername] = params[0]
            #best_parameters[learnername] = cross_validate(5, Xtrain, Ytrain, Learner, params)
            #best_parameters_scv[learnername] = stratified_cross_validate(5, Xtrain, Ytrain, Learner, params)
        print('Best parameters selected') 
        
        
        
        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            learner = Learner(params)
            learned_parameters = learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(Xtest, learned_parameters)
            error = geterror(Ytest,predictions)
            print('CV ', error)
            errors[learnername][r] = error

            params = best_parameters_scv[learnername]
            learner = Learner(params)
            learned_parameters = learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(Xtest, learned_parameters)
            error = geterror(Ytest,predictions)
            print('SCV ', error)
            errors_scv[learnername][r] = error
        
       
    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        aveerror_scv = np.mean(errors_scv[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerror))
        print('Average SCV error for ' + learnername + ': ' + str(aveerror_scv))    
