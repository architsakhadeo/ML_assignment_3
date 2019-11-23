import numpy as np
import math

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
def cross_validate(K, X, Y, learnername, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    length = len(X)
    batchlength = 1.0*length/K
    
    # Shuffles data indices
    
    indices = np.array([i for i in range(len(X))])
    shuffle_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(shuffle_indices)
    
    # Shuffles data with these shuffled indices
    
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]
    
    # Training and testing over mini batches
    
    for k in range(K):
        
        # Selects test data in batches
        
        testindices = np.array([i for i in range(int(k*batchlength),int((k+1)*batchlength))])
        
        # Selects train data = (all data - test data)
        
        trainindices = np.array([i for i in indices if i not in testindices])

        xtest = X[testindices]
        ytest = Y[testindices]
        xtrain = X[trainindices]
        ytrain = Y[trainindices]
        
        # Trains and tests on this data
         
        for i, params in enumerate(parameters):
            #print(k, i , params)
            learner = Algorithm(params)
            learned_parameters = learner.learn(xtrain, ytrain)
            predictions = learner.predict(xtest, learned_parameters)
            error = geterror(ytest,predictions)
            all_errors[i][k] = error

    # Calculates average error and standard error for every meta-parameter
    
    avg_errors = np.mean(all_errors, axis=1)
    std_errors = np.std(all_errors, axis=1)/math.sqrt(len(all_errors[0]))


    for i, params in enumerate(parameters):
        print('Cross validate parameters for ' + learnername + ' : ', params)
        print('Average error on cross validation data for ' + learnername + ' :', avg_errors[i])
        print('Standard error on cross validation data for ' + learnername + ' :', std_errors[i])
        print()
    best_parameters = parameters[np.argmin(avg_errors)]

    return best_parameters




# Stratified cross validation splits training and test data proportional to their labelled classes

def stratified_cross_validate(K, X, Y, learnername, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    batchlength = 1.0*len(X)/K

    # Shuffles data indices
    
    indices = np.array([i for i in range(len(X))])
    shuffle_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(shuffle_indices)
    
    group0_indices = np.array([])
    group1_indices = np.array([])

    # Selects indices of data with specific class
    
    for i in shuffle_indices:
        if Y[i] == 0:
            group0_indices = np.concatenate((group0_indices, [i]))
        if Y[i] == 1:
            group1_indices = np.concatenate((group1_indices, [i]))

    group0_indices = group0_indices.astype(int)
    group1_indices = group1_indices.astype(int)    
    
    # Calculates number of samples of each class
    # batchlength = group0_batchlength + group1_batchlength
    
    group0_batchlength = 1.0*len(group0_indices)/K
    group1_batchlength = 1.0*len(group1_indices)/K  

    final_shuffled_indices = np.array([])
    
    # Splits data into batches with proportion of classes equal to original data
    
    for k in range(K):
    
        temp_indices = np.array([])
        temp_indices = np.concatenate((temp_indices, group0_indices[int(k*group0_batchlength):int((k+1)*group0_batchlength)]))
        temp_indices = np.concatenate((temp_indices, group1_indices[int(k*group1_batchlength):int((k+1)*group1_batchlength)]))
        
        # Shuffles these data indices again to introduce randomness else the data with class 0 will be followed by all data with class 1

        np.random.shuffle(temp_indices)
        
        # Add these shuffled minibatch data indices together so each minibatch has same proportion of class 0 and class 1 data samples
        
        final_shuffled_indices = np.concatenate((final_shuffled_indices, temp_indices))
    
    final_shuffled_indices = final_shuffled_indices.astype(int)

    # Shuffles data with these shuffled indices
        
    X = X[final_shuffled_indices]
    Y = Y[final_shuffled_indices]
    
    # Training and testing over mini batches

    for k in range(K):
        
        # Selects test data in batches
        
        testindices = np.array([i for i in range(int(k*batchlength),int((k+1)*batchlength))])
        
        # Selects train data = (all data - test data)
        
        trainindices = np.array([i for i in indices if i not in testindices])

        xtest = X[testindices]
        ytest = Y[testindices]
        xtrain = X[trainindices]
        ytrain = Y[trainindices]
        
        # Trains and tests on this data
        
        for i, params in enumerate(parameters):
            #print(k, i , params)
            learner = Algorithm(params)
            learned_parameters = learner.learn(xtrain, ytrain)
            predictions = learner.predict(xtest, learned_parameters)
            error = geterror(ytest,predictions)
            all_errors[i][k] = error
    
    # Calculates average error and standard error for every meta-parameter
           
    avg_errors = np.mean(all_errors, axis=1)
    std_errors = np.std(all_errors, axis=1)/math.sqrt(len(all_errors[0]))

    for i, params in enumerate(parameters):
        print('Stratified cross validate parameters for ' + learnername + ' : ', params)
        print('Average error on stratified cross validation data for ' + learnername + ' :', avg_errors[i])
        print('Standard error on stratified cross validation data for ' + learnername + ' :', std_errors[i])
        print()
    best_parameters = parameters[np.argmin(avg_errors)]
    #print(best_parameters)
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
        'Naive Bayes': algs.NaiveBayes,
        'Linear Regression': algs.LinearRegressionClass,
        'Logistic Regression': algs.LogisticReg,
        'Neural Network 1 hidden layer': algs.NeuralNet_1hiddenlayer,
        'Neural Network 2 hidden layers': algs.NeuralNet_2hiddenlayers,
        'Linear Kernel Logistic Regression': algs.LinearKernelLogisticRegression,
        #'Hamming Distance Kernel Logistic Regression': algs.HammingDistanceKernelLogisticRegression,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Linear Regression': [
            {'regwgt': 0.01,'usecolumnones':True}, 
            {'regwgt': 0.01,'usecolumnones':False}
        ],
        'Naive Bayes': [
            # first set of parameters to try
            { 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Logistic Regression': [
            { 'epochs': 100, 'stepsize': 0.001 },
            { 'epochs': 100, 'stepsize': 0.005 },            
            { 'epochs': 100, 'stepsize': 0.01 },
            { 'epochs': 100, 'stepsize': 0.05 },            
        ],
        'Neural Network 1 hidden layer': [
            { 'epochs': 10, 'nh': 4 },
            { 'epochs': 10, 'nh': 8 },
            { 'epochs': 10, 'nh': 16 },
            { 'epochs': 10, 'nh': 32 },
        ],
        'Neural Network 2 hidden layers': [
            { 'epochs': 10, 'nh1': 4, 'nh2': 4 },
            { 'epochs': 10, 'nh1': 8, 'nh2': 8 },
            { 'epochs': 10, 'nh1': 16, 'nh2': 16 },
            { 'epochs': 10, 'nh1': 32, 'nh2': 32 },
        ],
        'Linear Kernel Logistic Regression': [
            { 'epochs': 20, 'centers': 10, 'stepsize': 0.01 },
            { 'epochs': 20, 'centers': 20, 'stepsize': 0.01 },
            { 'epochs': 20, 'centers': 40, 'stepsize': 0.01 },
            { 'epochs': 20, 'centers': 80, 'stepsize': 0.01 },
        ],
        'Hamming Distance Kernel Logistic Regression': [
            { 'epochs': 20, 'centers': 10, 'stepsize': 0.01 },
            { 'epochs': 20, 'centers': 20, 'stepsize': 0.01 },
            { 'epochs': 20, 'centers': 40, 'stepsize': 0.01 },
            { 'epochs': 20, 'centers': 80, 'stepsize': 0.01 },
        ]
    }

    # initialize the errors for each parameter setting to 0




    errors = {}
    errors_scv = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)
        errors_scv[learnername] = np.zeros(numruns)
        
    for r in range(numruns):
        print('\n----------')
        print('Run: ', r)
        print('----------\n')
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # Data preprocessing

        Xtrain = np.array(trainset[0])
        Ytrain = np.array(trainset[1])
        Xtest = np.array(testset[0])
        Ytest = np.array(testset[1])
       

        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])
        Ytest = np.reshape(Ytest, [len(Ytest), 1])
        a = Ytrain
        b = Ytest

                
        # cast the Y vector as a matrix
        
        X_temp = np.empty((0,len(Xtrain[0])))
        Y_temp = np.empty((0,len(Xtrain[0])))
                
        for i in range(len(Xtrain)):
            temp = np.array([])
            for j in range(len(Xtrain[i])):
                temp = np.concatenate((temp,[Xtrain[i][j]]))
            X_temp = np.append(X_temp,np.array([temp]), axis=0)
        
        for i in range(len(Ytrain)):
            Y_temp = np.append(Y_temp,np.array([Ytrain[i]]))
        
        
        Xtrain = X_temp
        Ytrain = Y_temp



        X_temp = np.empty((0,len(Xtest[0])))
        Y_temp = np.empty((0,len(Xtest[0])))

        for i in range(len(Xtest)):
            temp = np.array([])
            for j in range(len(Xtest[i])):
                temp = np.concatenate((temp,[Xtest[i][j]]))
            X_temp = np.append(X_temp,np.array([temp]), axis=0)
        
        for i in range(len(Ytest)):
            Y_temp = np.append(Y_temp,np.array([Ytest[i]]))
        
        
        Xtest = X_temp
        Ytest = Y_temp

        # cast the Y vector as a matrix       
        
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])        
        Ytest = np.reshape(Ytest, [len(Ytest), 1])


        best_parameters = {}
        best_parameters_scv = {}
        for learnername, Learner in classalgs.items():
            #print('Cross validation of ' + learnername)
            params = parameters.get(learnername, [ None ])
            
            # Change cross_validate to stratified_cross_validate
            
            best_parameters[learnername] = cross_validate(5, Xtrain, Ytrain, learnername, Learner, params)
            best_parameters_scv[learnername] = stratified_cross_validate(5, Xtrain, Ytrain, learnername, Learner, params)
            print('--------------------\n')
        
        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            learner = Learner(params)
            learned_parameters = learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(Xtest, learned_parameters)
            error = geterror(Ytest,predictions)
            print('Error on the actual test data for ' + learnername + ' for best cross validation parameter ' + str(best_parameters[learnername]) + ' : ' + str(error) + '\n')
            errors[learnername][r] = error
            
            params = best_parameters_scv[learnername]
            learner = Learner(params)
            learned_parameters = learner.learn(np.array(Xtrain), np.array(Ytrain))
            predictions = learner.predict(np.array(Xtest), learned_parameters)
            error = geterror(Ytest,predictions)
            print('Error on the actual test data for ' + learnername + ' for best stratified cross validation parameter ' + str(best_parameters[learnername]) + ' : ' + str(error) + '\n')
            errors_scv[learnername][r] = error
            print('--------------------\n')            
       
    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        stderror = np.std(errors[learnername])/math.sqrt(numruns)
        print('Average error on the actual test data for ' + learnername + ' after cross validation : ' + str(aveerror))
        print('Standard error for ' + learnername + ' after cross validation : ' + str(stderror))  
        print()
        aveerror_scv = np.mean(errors_scv[learnername])
        stderror_scv = np.std(errors_scv[learnername])/math.sqrt(numruns)
        print('Average error on the actual test data for ' + learnername + ' after stratified cross validation : ' + str(aveerror_scv))
        print('Standard error on the actual test data for ' + learnername + ' after stratified cross validation : ' + str(stderror_scv))
        print('\n--------------------\n')              
