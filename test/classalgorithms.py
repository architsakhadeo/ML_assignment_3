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
        self.params = utils.update_dictionary_items({'regwgt': 0.01, 'usecolumnones': True}, parameters)
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
        if self.params['usecolumnones'] == False:
            X = X[:,:-1]
        
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples
        #print(self.weights)


    def predict(self, Xtest, learned_parameters):
        if self.params['usecolumnones'] == False:
            Xtest = Xtest[:,:-1]
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
        
        # Calculates the prior probabilities of each class
        p_y_0 = (1.0*np.count_nonzero(ytrain==0))/len(ytrain)
        p_y_1 = (1.0*np.count_nonzero(ytrain==1))/len(ytrain)
        
        # Calculates mean and variance of all features with respect to their classes
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
        
        # Returns learned parameters which are means and variances per class and class prior probabilities
        learned_parameters = [mean_0, variance_0, mean_1, variance_1, p_y_0, p_y_1]
        return learned_parameters   


    def predict(self, Xtest, learned_parameters):
        if self.params['usecolumnones'] == False:
            Xtest = Xtest[:,:-1]            
        
        mean_0, variance_0, mean_1, variance_1, p_y_0, p_y_1 = learned_parameters

        numsamples = Xtest.shape[0]
        predictions = np.array([]) # p_yi_x as classes
        
        for i in range(len(Xtest)):
            
            # Calculates probabilities of every feature given class from a gaussian distribution over the features with given mean and variance
            
            #p_xij_given_yi_0 =((2*math.pi*variance_0)**(-0.5)) * math.e ** ((-(Xtest[i] - mean_0)**2) / (2 * variance_0))  
            #p_xij_given_yi_1 = ((2*math.pi*variance_1)**(-0.5)) * math.e ** ((-(Xtest[i] - mean_1)**2) / (2 * variance_1))
            
            
            
            # Uses utils gaussian distribution
            # Calculates probabilities of every feature given class from a gaussian distribution over the features with given mean and variance
            
            p_xij_given_yi_0 = np.zeros(Xtest.shape[1])
            p_xij_given_yi_1 = np.zeros(Xtest.shape[1])
            
            for j in range(Xtest.shape[1]):
                p_xij_given_yi_0[j] = utils.gaussian_pdf(Xtest[i][j],mean_0[j],variance_0[j]**0.5)
                p_xij_given_yi_1[j] = utils.gaussian_pdf(Xtest[i][j],mean_1[j],variance_1[j]**0.5)
            
            # Calculates probability of class given that data sample by multiplying feature probabilities given class by class prior probabilities     
                   
            p_yi_given_xi_0 = np.prod(p_xij_given_yi_0) * p_y_0  # approximation of Bayes' rule since denominator p(xi) is constant for both classes
            p_yi_given_xi_1 = np.prod(p_xij_given_yi_1) * p_y_1  # approximation of Bayes' rule since denominator p(xi) is constant for both classes
            
            # Compares probabilities of classes given data sample
            
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
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100, 'regwgt': 0.5}, parameters)
        self.weights = None


    def learn(self, X, y):
        
        Xtrain = X
        ytrain = y
        
        self.weights = np.random.rand(Xtrain.shape[1])
        
        # Logistic regression with Stochastic Gradient Descent
        
        epochs = self.params['epochs']
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        
        for epoch in range(epochs):
            #print(epoch)
            np.random.shuffle(shuffle_indices)
            for index in shuffle_indices:
                delta_c_index = (Xtrain[index] * ((utils.sigmoid((Xtrain[index].T).dot(self.weights))) - ytrain[index]) )# + 2*self.params['regwgt']/(Xtrain.shape[0])*np.linalg.norm(self.weights) # for regularization
                self.weights = self.weights - self.params['stepsize'] * delta_c_index
        
        # Returns weights as learned_parameters
        return self.weights
        

    def predict(self, Xtest, learned_parameters):
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        weights = learned_parameters
        
        
        # Thresholding of probabilities into classes
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
        self.bi = np.random.randn(self.params['nh'], 1)
        self.bo = np.random.randn(ytrain.shape[1], 1)

        wi = self.wi
        wo = self.wo
        bi = self.bi
        bo = self.bo

        # Xtrain is n x d,Xtrain[index] is 1 x d, Xtrain[index].T is d x 1
        
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        epochs = self.params['epochs']
        
        for epoch in range(epochs):
            E = 0
            np.random.shuffle(shuffle_indices)
            #print(epoch)
            for index in shuffle_indices:
                
                # Selects one data sample for Stochastic gradient descent update 
                
                X = np.reshape(Xtrain[index], [len(Xtrain[index]),1])
                Y = np.reshape(ytrain[index], [1,1])
                
                # Forward propagation               
                
                Z2 = np.dot(wi, X) + bi
                A2 = utils.sigmoid(Z2)         
                Z1 = np.dot(wo, A2) + bo
                A1 = utils.sigmoid(Z1)                
                
                # Neural network loss calculation
                e = - np.dot(Y, np.log(A1)) - np.dot(1-Y,np.log(1-A1))
                E += e
                
                # Gradient calculations with respect to each weight and bias
                
                dE_A1 = - np.divide(Y,A1) + np.divide(1-Y, 1-A1)
                dE_Z1 = dE_A1 * utils.dsigmoid(Z1)
                
                dE_wo = np.dot(dE_Z1, A2.T)
                dE_bo = np.dot(dE_Z1, np.ones([dE_Z1.shape[1],1]))

                dE_A2 = np.dot(wo.T, dE_Z1)                
                dE_Z2 = dE_A2 * utils.dsigmoid(Z2)

                dE_wi = np.dot(dE_Z2, X.T)
                dE_bi = np.dot(dE_Z2, np.ones([dE_Z2.shape[1],1]))
                
                # Back propagation using 
                
                wo = wo - self.params['stepsize'] * dE_wo
                bo = bo - self.params['stepsize'] * dE_bo
                wi = wi - self.params['stepsize'] * dE_wi
                bi = bi - self.params['stepsize'] * dE_bi

            
            #print(E)
        
        # Returns learned parameters
        
        learned_parameters = [wo, bo, wi, bi]
        return learned_parameters
 

    def predict(self,Xtest, learned_parameters):
        
        wo, bo, wi, bi = learned_parameters
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        # Tests every data sample separately
                
        for index in range(Xtest.shape[0]):
        
            X = np.reshape(Xtest[index], [len(Xtest[index]),1])
            
            # Forward propagation over test data for the learned weights and bias
            
            Z2 = np.dot(wi, X) + bi
            A2 = utils.sigmoid(Z2)                
            Z1 = np.dot(wo, A2) + bo
            A1 = utils.sigmoid(Z1)
            
            # Thresholding sigmoid output into classes
        
            if A1 > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (A1) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same        
        
        return np.reshape(predictions, [numsamples, 1])


    def evaluate(self, inputs):
        pass


    def update(self, inputs, outputs):
        pass








# Implements Adam and not RMSProp updates. We had not implemented RMSPRop in the previous assignment. In a discussion on eClass
# the TA mentioned to implement either Adam or RMSProp.

class NeuralNet_2hiddenlayers(Classifier):


    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh1': 4,
            'nh2': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.005,
            'epochs': 10,
            'momentum_coeff1': 0.9,
            'momentum_coeff2': 0.999, 
            'epsilon': 1e-8
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
        self.bi = np.random.randn(self.params['nh1'], 1)
        self.bm = np.random.randn(self.params['nh2'], 1)
        self.bo = np.random.randn(ytrain.shape[1], 1)
        
        wi = self.wi
        wm = self.wm
        wo = self.wo
        bi = self.bi
        bm = self.bm
        bo = self.bo
        
        M_wo = 0
        V_wo = 0
        M_bo = 0
        V_bo = 0
        M_wm = 0
        V_wm = 0
        M_bm = 0
        V_bm = 0
        M_wi = 0
        V_wi = 0
        M_bi = 0
        V_bi = 0
        
        
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        epochs = self.params['epochs']
        
        for epoch in range(epochs):
            E = 0
            np.random.shuffle(shuffle_indices)
            #print(epoch)
            for index in shuffle_indices:
                
                # Selects one data sample for Adam update
                
                X = np.reshape(Xtrain[index], [len(Xtrain[index]),1])
                Y = np.reshape(ytrain[index], [1,1])
                
                # Forward propagation
                
                Z3 = np.dot(wi, X) + bi
                A3 = utils.sigmoid(Z3)         
                Z2 = np.dot(wm, A3) + bm
                A2 = utils.sigmoid(Z2)
                Z1 = np.dot(wo, A2) + bo
                A1 = utils.sigmoid(Z1)                
                
                # Neural network loss calculation
                
                e = - np.dot(Y, np.log(A1)) - np.dot(1-Y,np.log(1-A1))
                E += e
                
                # Gradient calculations with respect to each weight and bias
                
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
                
                # Adam update from previous assignment, calculation of moments and squared gradients
                
                # Implements Adam and not RMSProp updates. We had not implemented RMSPRop in the previous assignment. In a discussion on eClass
                # the TA mentioned to implement either Adam or RMSProp.

                M_wo = self.params['momentum_coeff1'] * M_wo + (1-self.params['momentum_coeff1'])*dE_wo
                V_wo = self.params['momentum_coeff2'] * V_wo + (1-self.params['momentum_coeff2'])*(np.multiply(dE_wo ,dE_wo))
                M_bo = self.params['momentum_coeff1'] * M_bo + (1-self.params['momentum_coeff1'])*dE_bo
                V_bo = self.params['momentum_coeff2'] * V_bo + (1-self.params['momentum_coeff2'])*(np.multiply(dE_bo ,dE_bo))
                M_wm = self.params['momentum_coeff1'] * M_wm + (1-self.params['momentum_coeff1'])*dE_wm
                V_wm = self.params['momentum_coeff2'] * V_wm + (1-self.params['momentum_coeff2'])*(np.multiply(dE_wm ,dE_wm))
                M_bm = self.params['momentum_coeff1'] * M_bm + (1-self.params['momentum_coeff1'])*dE_bm
                V_bm = self.params['momentum_coeff2'] * V_bm + (1-self.params['momentum_coeff2'])*(np.multiply(dE_bm ,dE_bm))                                
                M_wi = self.params['momentum_coeff1'] * M_wi + (1-self.params['momentum_coeff1'])*dE_wi
                V_wi = self.params['momentum_coeff2'] * V_wi + (1-self.params['momentum_coeff2'])*(np.multiply(dE_wi ,dE_wi))
                M_bi = self.params['momentum_coeff1'] * M_bi + (1-self.params['momentum_coeff1'])*dE_bi
                V_bi = self.params['momentum_coeff2'] * V_bi + (1-self.params['momentum_coeff2'])*(np.multiply(dE_bi ,dE_bi))
                
                # Back propagation
                
                wo = wo - self.params['stepsize'] * M_wo / (np.sqrt(V_wo) + self.params['epsilon'])
                bo = bo - self.params['stepsize'] * M_bo / (np.sqrt(V_bo) + self.params['epsilon'])
                wm = wm - self.params['stepsize'] * M_wm / (np.sqrt(V_wm) + self.params['epsilon'])
                bm = bm - self.params['stepsize'] * M_bm / (np.sqrt(V_bm) + self.params['epsilon'])
                wi = wi - self.params['stepsize'] * M_wi / (np.sqrt(V_wi) + self.params['epsilon'])
                bi = bi - self.params['stepsize'] * M_bi / (np.sqrt(V_bi) + self.params['epsilon'])
                        
        learned_parameters = [wo, bo, wm, bm, wi, bi]
        return learned_parameters


    def predict(self,Xtest, learned_parameters):
        wo, bo, wm, bm, wi, bi = learned_parameters
        
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        # Tests every data sample separately
        
        for index in range(Xtest.shape[0]):
        
            X = np.reshape(Xtest[index], [len(Xtest[index]),1])
            
            # Forward propagation over test data for the learned weights and bias
            
            Z3 = np.dot(wi, X) + bi
            A3 = utils.sigmoid(Z3)         
            Z2 = np.dot(wm, A3) + bm
            A2 = utils.sigmoid(Z2)
            Z1 = np.dot(wo, A2) + bo
            A1 = utils.sigmoid(Z1)       
        
            # Thresholding sigmoid output into classes
            
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
class LinearKernelLogisticRegression(LogisticReg):


    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
            'regwgt': 0.5
        }, parameters)
        self.weights = None


    def learn(self, X, y):
        
        Xtrain = X
        ytrain = y
        
        # Selects centers randomly from the Xtrain data
        
        random_indices = np.random.choice(range(len(Xtrain)), self.params['centers'], replace = False)
        centers = Xtrain[random_indices]
        
        # Linear kernel that transforms Xtrain (n x d) to a new matrix (n x c) dimensions by dot product with centers.
        
        Xtrain = np.dot(Xtrain, centers.T)
        self.weights = np.random.rand(Xtrain.shape[1])
        
        # Linear Kernel Logistic Regression with Stochastic Gradient Descent
        
        epochs = self.params['epochs']
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        
        for epoch in range(epochs):
            #print(epoch)
            np.random.shuffle(shuffle_indices)
            for index in shuffle_indices:
                delta_c_index = Xtrain[index] * ((utils.sigmoid((Xtrain[index].T).dot(self.weights))) - ytrain[index]) #+ 2*self.params['regwgt']/(Xtrain.shape[0])*np.linalg.norm(self.weights) # for regularization
                self.weights = self.weights - self.params['stepsize'] * delta_c_index
        
        learned_parameters = [self.weights, centers]
        return learned_parameters


    def predict(self, Xtest, learned_parameters):
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        weights, centers = learned_parameters
        
        # Applies kernel function on test data
        
        Xtest = np.dot(Xtest, centers.T)
        
        # Thresholding sigmoid output into classes
        
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
            'regwgt': 0.5
        }, parameters)
        self.weights = None


    def learn(self, X, y):
        
        Xtrain = X
        ytrain = y
        # Selects centers randomly from the Xtrain data
        
        random_indices = np.random.choice(range(len(Xtrain)), self.params['centers'], replace = False)
        centers = Xtrain[random_indices]
        
        # Hamming distance kernel that transforms Xtrain (n x d) to a new matrix (n x c) dimensions by finding Hamming distance with original data
        # Hamming distance of a feature will be 1 if the same feature is different between two samples. Hamming distance of two samples is the sum
        # of hamming distances over all of their features
        
        X_C = np.empty((0,self.params['centers']))
        for x in range(Xtrain.shape[0]):
            temp = np.array([])
            #print(Xtrain[x])
            for c in range(len(centers)):
                temp = np.concatenate((temp, [np.sum(Xtrain[x] != centers[c])] ))
            X_C = np.append(X_C, np.array([temp]), axis=0)
        
        Xtrain = X_C


        #for i in range(Xtrain.shape[0]):
        #    print(Xtrain[i])
            
        self.weights = np.random.rand(Xtrain.shape[1])
        
        # Hamming Distance Kernel Logistic Regression with Stochastic Gradient Descent
        
        epochs = self.params['epochs']
        shuffle_indices = np.array(range(Xtrain.shape[0]))
        
        for epoch in range(epochs):
            #print(epoch)
            np.random.shuffle(shuffle_indices)
            for index in shuffle_indices:
                delta_c_index = Xtrain[index] * ((utils.sigmoid((Xtrain[index].T).dot(self.weights))) - ytrain[index]) #+ 2*self.params['regwgt']/(Xtrain.shape[0])*np.linalg.norm(self.weights) # for regularization
                self.weights = self.weights - self.params['stepsize'] * delta_c_index
        
        
        learned_parameters = [self.weights, centers]
        return learned_parameters


    def predict(self, Xtest, learned_parameters):
        predictions = np.array([])
        numsamples = Xtest.shape[0]
        
        weights, centers = learned_parameters
        
        # Applies kernel function on test data
        
        X_C = np.empty((0,self.params['centers']))
        for x in range(Xtest.shape[0]):
            temp = np.array([])
            for c in range(len(centers)):
                temp = np.concatenate((temp, [np.sum(Xtest[x] != centers[c])] ))
            X_C = np.append(X_C, np.array([temp]), axis=0)
        
        Xtest = X_C

        #for i in range(Xtest.shape[0]):
        #    print(Xtest[i])        
        
        # Thresholding sigmoid output into classes
        
        for index in range(Xtest.shape[0]):
            if (utils.sigmoid((Xtest[index].T).dot(self.weights))) > 0.5:
                predictions = np.concatenate((predictions, [1] ))
            elif (utils.sigmoid((Xtest[index].T).dot(self.weights))) < 0.5:
                predictions = np.concatenate((predictions, [0] ))
            else:
                predictions = np.concatenate((predictions, [np.random.randint(2)] )) # either 0 or 1 arbitrarily if both probabilities are same
        
        return np.reshape(predictions, [numsamples, 1])


