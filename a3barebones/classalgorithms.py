import numpy as np
import math
import MLCourse.utilities as utils

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest, learned_parameters):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        
        #X = X[:,:-1]
        
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest, learned_parameters):
        
        #Xtest = Xtest[:,:-1]
    
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items({'usecolumnones': False}, parameters)
        self.mean_0 = []
        self.mean_1 = []
        self.variance_0 = []
        self.variance_1 = []

    def learn(self, Xtrain, ytrain):
        if self.params['usecolumnones'] == False:
            Xtrain = Xtrain[:,:-1]
            
    
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')
        
        p_y_0 = (1.0*np.count_nonzero(ytrain==0))/len(ytrain)
        p_y_1 = (1.0*np.count_nonzero(ytrain==1))/len(ytrain)

        
        mean_0 = np.zeros(Xtrain.shape[1])
        mean_1 = np.zeros(Xtrain.shape[1])
        m0 = 0
        m1 = 0
        for i in range(len(Xtrain)):
            if ytrain[i]==0:
                mean_0 = np.add(mean_0,Xtrain[i])
                m0 += 1
            if ytrain[i]==1:
                mean_1 = np.add(mean_1,Xtrain[i])
                m1 += 1
        mean_0 = (1.0 * mean_0) / m0
        mean_1 = (1.0 * mean_1) / m1

        variance_0 = np.zeros(Xtrain.shape[1])
        variance_1 = np.zeros(Xtrain.shape[1])
        for i in range(len(Xtrain)):
            if ytrain[i]==0:
                variance_0 = np.add(variance_0,(Xtrain[i] - mean_0)**2)
            if ytrain[i]==1:
                variance_1 = np.add(variance_1,(Xtrain[i] - mean_1)**2)
        variance_0 = (1.0 * variance_0) / m0
        variance_1 = (1.0 * variance_1) / m1

        learned_parameters = [mean_0, variance_0, mean_1, variance_1, p_y_0, p_y_1]
        return learned_parameters   

    def predict(self, Xtest, learned_parameters):
        
        if self.params['usecolumnones'] == False:
            Xtest = Xtest[:,:-1]            
        
        mean_0, variance_0, mean_1, variance_1, p_y_0, p_y_1 = learned_parameters

        numssample = Xtest.shape[0]
        predictions = np.array([]) # p_yi_x
        
        for i in range(len(Xtest)):
            p_xij_given_yi_0 = np.zeros(Xtest.shape[1])
            p_xij_given_yi_1 = np.zeros(Xtest.shape[1])
            
            p_xij_given_yi_0 = ((2*math.pi*variance_0)**(-0.5)) * math.e ** ((-(Xtest[i] - mean_0)**2) / (2 * variance_0))  
            p_xij_given_yi_1 = ((2*math.pi*variance_1)**(-0.5)) * math.e ** ((-(Xtest[i] - mean_1)**2) / (2 * variance_1))
            #for j in range(len(Xtest[i])):
                #p_xij_given_yi_0 = np.concatenate( (p_xij_given_yi_0,( ((2*math.pi*variance_0[j])**0.5) * math.e ** (-(Xtest[i][j] - mean_0[j])**2) / (2 * variance_0[j]) )))
                #p_xij_given_yi_1 = np.concatenate( (p_xij_given_yi_1,( ((2*math.pi*variance_1[j])**0.5) * math.e ** (-((Xtest[i][j] - mean_1[j])**2) / (2 * variance_1[j]) )))
                
            p_yi_given_xi_0 = np.prod(p_xij_given_yi_0) * p_y_0  # approximation of Bayes' rule since denominator p(xi) is constant for both classes
            p_yi_given_xi_1 = np.prod(p_xij_given_yi_1) * p_y_1  # approximation of Bayes' rule since denominator p(xi) is constant for both classes
            if p_yi_given_xi_0 > p_yi_given_xi_1:
                predictions = np.concatenate((predictions, [0] ))
            elif p_yi_given_xi_0 < p_yi_given_xi_1:
                predictions = np.concatenate((predictions, [1] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same
                
        return np.reshape(predictions, [numsamples, 1])




# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None

    def learn(self, X, y):
        # Assume X as 1Xd
        
        Xtrain = X
        ytrain = y
        
        self.weights = np.random.rand(Xtrain.shape[1])
        
        # Stochastic Gradient Descent
        
        epochs = self.params['epochs']
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        
        for epoch in range(epochs):
            #print(epoch)
            np.random.shuffle(shuffle_indices)
            for index in shuffle_indices:
                delta_c_index = Xtrain[index] * ((utils.sigmoid((Xtrain[index].T).dot(self.weights))) - ytrain[index])
                self.weights = self.weights - self.params['stepsize'] * delta_c_index
        
        return self.weights
        
    def predict(self, Xtest, learned_parameters):
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        weights = learned_parameters
        
        for index in range(Xtest.shape[0]):
            if (utils.sigmoid((Xtest[index].T).dot(self.weights))) > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (utils.sigmoid((Xtest[index].T).dot(self.weights))) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same
        
        return np.reshape(predictions, [numsamples, 1])




# Susy: ~23 error (4 hidden units)
class NeuralNet_1hiddenlayer(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None
        self.bi = None
        self.bo = None

    def learn(self, Xtrain, ytrain):
        self.wi = np.random.randn(self.params['nh'], Xtrain.shape[1])
        self.wo = np.random.randn(ytrain.shape[1], self.params['nh'])
        wi = self.wi
        wo = self.wo
        bi = self.bi
        bo = self.bo
        
        
        bi = np.random.randn(self.params['nh'], 1)
        bo = np.random.randn(ytrain.shape[1], 1)
        
        # Xtrain is nXd,Xtrain[index] is 1Xd, Xtrain[index].T is dX1
        
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        epochs = self.params['epochs']
        
        for epoch in range(epochs):
            E = 0
            np.random.shuffle(shuffle_indices)
            print(epoch)
            for index in shuffle_indices:

                X = np.reshape(Xtrain[index], [len(Xtrain[index]),1])
                Y = np.reshape(ytrain[index], [1,1])
                               
                Z2 = np.dot(wi, X) + bi
                A2 = utils.sigmoid(Z2)         
                Z1 = np.dot(wo, A2) + bo
                A1 = utils.sigmoid(Z1)                
                
                #print(Z2, A2, Z1, A1)

                e = - np.dot(Y, np.log(A1)) - np.dot(1-Y,np.log(1-A1))
                E += e
                
                dE_A1 = - np.divide(Y,A1) + np.divide(1-Y, 1-A1)
                dE_Z1 = dE_A1 * utils.dsigmoid(Z1)
                
                dE_wo = np.dot(dE_Z1, A2.T)
                dE_bo = np.dot(dE_Z1, np.ones([dE_Z1.shape[1],1]))

                dE_A2 = np.dot(wo.T, dE_Z1)                
                dE_Z2 = dE_A2 * utils.dsigmoid(Z2)

                dE_wi = np.dot(dE_Z2, X.T)
                dE_bi = np.dot(dE_Z2, np.ones([dE_Z2.shape[1],1]))
                                
                wo = wo - self.params['stepsize'] * dE_wo
                bo = bo - self.params['stepsize'] * dE_bo
                wi = wi - self.params['stepsize'] * dE_wi
                bi = bi - self.params['stepsize'] * dE_bi

            
            print(E)
                        
        learned_parameters = [wo, bo, wi, bi]
        return learned_parameters
        '''
        # Stochastic Gradient Descent
        
        # E = 1X1
        # y = 1X1
        # A1 = 1X1
        # Z1 = 1X1
        # w0 = 1Xh
        # A2 = hX1
        # Z2 = hX1
        # wi = hXd
        # X = dX1
        
        # A1 = sigmoid(Z1)
        # Z1 = wo.A2
        # A2 = sigmoid(Z2) element wise
        # Z2 = wi.X
        
        
        for epoch in range(self.params['epochs']):
            np.random.shuffle(shuffle_indices)
            print(epoch)
            for index in shuffle_indices:
                
                # feedforward propagation
                
                Z2 = np.dot(wi, Xtrain[index].T)
                Z2 = np.reshape(Z2, [len(wi),1])
                
                A2 = utils.sigmoid(Z2)
                
                Z1 = np.dot(wo, A2)
                Z1 = np.reshape(Z1, [1,1])
                
                A1 = utils.sigmoid(Z1)
                
                #E = - (np.dot(ytrain[index], np.log(A1) + np.dot(1-ytrain[index],np.log(1-A1) ) ) # cost
                
                delta_wo = - ( (1-A1)*np.reshape(ytrain[index],[1,1]) - (A1)*(1-np.reshape(ytrain[index],[1,1])) ) * np.reshape(A2, [1,len(A2)])# A2.T  # scalar * 1Xh
                delta_wi = - wo.T * (delta_wo.dot(1 - A2)) * (np.reshape(Xtrain[index], [1,len(Xtrain[index])]))
                
                wo = wo - self.params['stepsize']*delta_wo
                wi = wi - self.params['stepsize']*delta_wi
        
        learned_parameters = [wo, wi]
        return learned_parameters
        '''        

    def predict(self,Xtest, learned_parameters):
        wo, bo, wi, bi = learned_parameters
        
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
                
        for index in range(Xtest.shape[0]):
            X = np.reshape(Xtest[index], [len(Xtest[index]),1])
            
            Z2 = np.dot(wi, X) + bi
            A2 = utils.sigmoid(Z2)                
            Z1 = np.dot(wo, A2) + bo
            A1 = utils.sigmoid(Z1)
        
            if A1 > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (A1) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same        
        
        return np.reshape(predictions, [numsamples, 1])

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, nh]
        )

    def update(self, inputs, outputs):
        pass




class NeuralNet_2hiddenlayers(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh1': 4,
            'nh2': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.005,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wm = None
        self.wo = None
        self.bi = None
        self.bm = None
        self.bo = None

    def learn(self, Xtrain, ytrain):
        self.wi = np.random.randn(self.params['nh1'], Xtrain.shape[1])
        self.wm = np.random.randn(self.params['nh2'], self.params['nh1'])
        self.wo = np.random.randn(ytrain.shape[1], self.params['nh2'])
        wi = self.wi
        wm = self.wm
        wo = self.wo
        bi = self.bi
        bm = self.bm
        bo = self.bo
        
        
        bi = np.random.randn(self.params['nh1'], 1)
        bm = np.random.randn(self.params['nh2'], 1)
        bo = np.random.randn(ytrain.shape[1], 1)
        
        # Xtrain is nXd,Xtrain[index] is 1Xd, Xtrain[index].T is dX1
        
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        epochs = self.params['epochs']
        
        for epoch in range(epochs):
            E = 0
            np.random.shuffle(shuffle_indices)
            print(epoch)
            for index in shuffle_indices:

                X = np.reshape(Xtrain[index], [len(Xtrain[index]),1])
                Y = np.reshape(ytrain[index], [1,1])
                               
                Z3 = np.dot(wi, X) + bi
                A3 = utils.sigmoid(Z3)         
                Z2 = np.dot(wm, A3) + bm
                A2 = utils.sigmoid(Z2)
                Z1 = np.dot(wo, A2) + bo
                A1 = utils.sigmoid(Z1)                
                
                #print(Z2, A2, Z1, A1)

                e = - np.dot(Y, np.log(A1)) - np.dot(1-Y,np.log(1-A1))
                E += e
                
                dE_A1 = - np.divide(Y,A1) + np.divide(1-Y, 1-A1)
                dE_Z1 = dE_A1 * utils.dsigmoid(Z1)

                dE_wo = np.dot(dE_Z1, A2.T)
                dE_bo = np.dot(dE_Z1, np.ones([dE_Z1.shape[1],1]))
                
                dE_A2 = np.dot(wo.T, dE_Z1)
                dE_Z2 = dE_A2 * utils.dsigmoid(Z2)
                
                dE_wm = np.dot(dE_Z2, A3.T)
                dE_bm = np.dot(dE_Z2, np.ones([dE_Z2.shape[1],1]))
                
                dE_A3 = np.dot(wm.T, dE_Z2)
                dE_Z3 = dE_A3 * utils.dsigmoid(Z3)
                
                dE_wi = np.dot(dE_Z3, X.T)
                dE_bi = np.dot(dE_Z3, np.ones([dE_Z3.shape[1], 1]))
                                
                wo = wo - self.params['stepsize'] * dE_wo
                bo = bo - self.params['stepsize'] * dE_bo
                wm = wm - self.params['stepsize'] * dE_wm
                bm = bm - self.params['stepsize'] * dE_bm
                wi = wi - self.params['stepsize'] * dE_wi
                bi = bi - self.params['stepsize'] * dE_bi

            
            print(E)
                        
        learned_parameters = [wo, bo, wm, bm, wi, bi]
        return learned_parameters
        '''
        # Stochastic Gradient Descent
        
        # E = 1X1
        # y = 1X1
        # A1 = 1X1
        # Z1 = 1X1
        # w0 = 1Xh
        # A2 = hX1
        # Z2 = hX1
        # wi = hXd
        # X = dX1
        
        # A1 = sigmoid(Z1)
        # Z1 = wo.A2
        # A2 = sigmoid(Z2) element wise
        # Z2 = wi.X
        
        
        for epoch in range(self.params['epochs']):
            np.random.shuffle(shuffle_indices)
            print(epoch)
            for index in shuffle_indices:
                
                # feedforward propagation
                
                Z2 = np.dot(wi, Xtrain[index].T)
                Z2 = np.reshape(Z2, [len(wi),1])
                
                A2 = utils.sigmoid(Z2)
                
                Z1 = np.dot(wo, A2)
                Z1 = np.reshape(Z1, [1,1])
                
                A1 = utils.sigmoid(Z1)
                
                #E = - (np.dot(ytrain[index], np.log(A1) + np.dot(1-ytrain[index],np.log(1-A1) ) ) # cost
                
                delta_wo = - ( (1-A1)*np.reshape(ytrain[index],[1,1]) - (A1)*(1-np.reshape(ytrain[index],[1,1])) ) * np.reshape(A2, [1,len(A2)])# A2.T  # scalar * 1Xh
                delta_wi = - wo.T * (delta_wo.dot(1 - A2)) * (np.reshape(Xtrain[index], [1,len(Xtrain[index])]))
                
                wo = wo - self.params['stepsize']*delta_wo
                wi = wi - self.params['stepsize']*delta_wi
        
        learned_parameters = [wo, wi]
        return learned_parameters
        '''        

    def predict(self,Xtest, learned_parameters):
        wo, bo, wm, bm, wi, bi = learned_parameters
        
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
                
        for index in range(Xtest.shape[0]):
            X = np.reshape(Xtest[index], [len(Xtest[index]),1])
            
            Z3 = np.dot(wi, X) + bi
            A3 = utils.sigmoid(Z3)         
            Z2 = np.dot(wm, A3) + bm
            A2 = utils.sigmoid(Z2)
            Z1 = np.dot(wo, A2) + bo
            A1 = utils.sigmoid(Z1)       
        
            if A1 > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (A1) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same        
        
        return np.reshape(predictions, [numsamples, 1])

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, nh]
        )

    def update(self, inputs, outputs):
        pass







# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        # Assume X as 1Xd
        
        Xtrain = X
        ytrain = y
        
        '''
        # Find centers with mean and variance of Xtrain
        
        mean = np.zeros(Xtrain.shape[1])
        m = 0
        for i in range(len(Xtrain)):
            mean = np.add(mean,Xtrain[i])
            m += 1
        mean = (1.0 * mean) / m

        variance = np.zeros(Xtrain.shape[1])
        for i in range(len(Xtrain)):
            variance = np.add(variance,(Xtrain[i] - mean)**2)
        variance = (1.0 * variance) / m
        '''

        random_indices = np.random.choice(range(len(Xtrain)), self.params['centers'], replace = False)
        centers = Xtrain[random_indices]
        
        Xtrain = np.dot(Xtrain, centers.T)
        self.weights = np.random.rand(Xtrain.shape[1])
        
        # Stochastic Gradient Descent
        
        epochs = self.params['epochs']
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        
        for epoch in range(epochs):
            #print(epoch)
            np.random.shuffle(shuffle_indices)
            for index in shuffle_indices:
                delta_c_index = Xtrain[index] * ((utils.sigmoid((Xtrain[index].T).dot(self.weights))) - ytrain[index])
                self.weights = self.weights - self.params['stepsize'] * delta_c_index
        
        learned_parameters = [self.weights, centers]
        return learned_parameters


    def predict(self, Xtest, learned_parameters):
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        weights, centers = learned_parameters
        Xtest = np.dot(Xtest, centers.T)
        
        
        for index in range(Xtest.shape[0]):
            if (utils.sigmoid((Xtest[index].T).dot(self.weights))) > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (utils.sigmoid((Xtest[index].T).dot(self.weights))) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same
        
        return np.reshape(predictions, [numsamples, 1])


class HammingDistanceKernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        # Assume X as 1Xd
        
        Xtrain = X
        ytrain = y
        
        '''
        # Find centers with mean and variance of Xtrain
        
        mean = np.zeros(Xtrain.shape[1])
        m = 0
        for i in range(len(Xtrain)):
            mean = np.add(mean,Xtrain[i])
            m += 1
        mean = (1.0 * mean) / m

        variance = np.zeros(Xtrain.shape[1])
        for i in range(len(Xtrain)):
            variance = np.add(variance,(Xtrain[i] - mean)**2)
        variance = (1.0 * variance) / m
        '''

        random_indices = np.random.choice(range(len(Xtrain)), self.params['centers'], replace = False)
        centers = Xtrain[random_indices]
        
        X_C = np.empty((0,self.params['centers']))
        for x in range(Xtrain.shape[0]):
            temp = np.array([])
            for c in range(len(centers)):
                temp = np.concatenate((temp, [np.sum(Xtrain[x] != centers[c])] ))
            X_C = np.append(X_C, np.array([temp]), axis=0)
        
        Xtrain = X_C
        print(Xtrain)


        self.weights = np.random.rand(Xtrain.shape[1])
        
        # Stochastic Gradient Descent
        
        epochs = self.params['epochs']
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        
        for epoch in range(epochs):
            #print(epoch)
            np.random.shuffle(shuffle_indices)
            for index in shuffle_indices:
                delta_c_index = Xtrain[index] * ((utils.sigmoid((Xtrain[index].T).dot(self.weights))) - ytrain[index])
                self.weights = self.weights - self.params['stepsize'] * delta_c_index
        
        learned_parameters = [self.weights, centers]
        return learned_parameters


    def predict(self, Xtest, learned_parameters):
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        weights, centers = learned_parameters


        X_C = np.empty((0,self.params['centers']))
        for x in range(Xtest.shape[0]):
            temp = np.array([])
            for c in range(len(centers)):
                temp = np.concatenate((temp, [np.sum(Xtest[x] != centers[c])] ))
            X_C = np.append(X_C, np.array([temp]), axis=0)
        
        Xtest = X_C
        
        
        for index in range(Xtest.shape[0]):
            if (utils.sigmoid((Xtest[index].T).dot(self.weights))) > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (utils.sigmoid((Xtest[index].T).dot(self.weights))) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same
        
        return np.reshape(predictions, [numsamples, 1])


