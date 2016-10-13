import numpy as np
import time
import cPickle, string, getopt, sys, random, re, pprint
import onlineldavb
import wikirandom

from robo.task.base_task import BaseTask


class LDA(BaseTask):

    def __init__(self, train=None, train_targets=None,
                 valid=None, valid_targets=None,
                 test=None, test_targets=None,
                 n_classes=None, num_epochs=500,
                 save=False, file_name=None):
        '''
        Logistic Regression. This benchmark
        consists of 5 different hyperparamters:
            Learning rate
            L2 regularisation
            Batch size
            Dropout rate
            L1 regularisation

        Parameters
        ----------
        train : (N, D) numpy array
            Training matrix where N are the number of data points
            and D are the number of features
        train_targets : (N) numpy array
            Labels for the training data
        valid : (K, D) numpy array
            Validation data
        valid_targets : (K) numpy array
            Validation labels
        test : (L, D) numpy array
            Test data
        test_targets : (L) numpy array
            Test labels
        n_classes: int
            Number of classes in the dataset
        '''
        self.X_train = train
        self.y_train = train_targets
        self.X_val = valid
        self.y_val = valid_targets
        self.X_test = test
        self.y_test = test_targets
        self.num_epochs = num_epochs
        self.save = save
        self.file_name = file_name
        # 1 Dim Number of topics: 2 to 100
        # 2 Dim Dirichlet distribution prior base measure alpha
        # 3 Dim Dirichlet distribution prior base measure eta
        # 4 Dim Learning rate
        # 5 Dim Weight Decay
        X_lower = np.array([2, 0, 0., 0.1, 1e-5])
        X_upper = np.array([100., 2., 2., 1., 1.])

        super(LDA, self).__init__(X_lower, X_upper)




    def set_weights(self, old_file_name):
        file_name = old_file_name + '.pkl'
        with open(file_name, 'rb') as f:
            param_values = cPickle.load(f)
        
        # self.olda._rhot = param_values[0]
        # self.olda._lambda = param_values[1]
        # self.olda._Elogbeta = param_values[2]
        # self.olda._expElogbeta = param_values[3]
        # self.olda._updatect = param_values[4]
        self._rhot = param_values[0]
        self._lambda = param_values[1]
        self._Elogbeta = param_values[2]
        self._expElogbeta = param_values[3]
        self._updatect = param_values[4]

    def set_epochs(self, n_epochs):
        self.num_epochs = n_epochs

    def set_save_modus(self, is_old=True, file_old=None, file_new=None):
        self.save_old = is_old
        self.save = True
        if self.save_old:
            self.file_name = file_old
        else:
            self.file_name = file_new


    def objective_function(self, x):
        # 1 Dim Number of topics: 2 to 100
        # 2 Dim Dirichlet distribution prior base measure alpha
        # 3 Dim Dirichlet distribution prior base measure eta
        # 4 Dim Learning rate
        # 5 Dim Weight Decay        
        topics_number = np.float32(np.exp(x[0, 0]))
        dirichlet_alpha= np.float32(x[0, 1])
        dirichlet_eta = np.int32(x[0, 2])
        weight_decay = np.int32(x[0, 3])
        l1_reg = np.int32(x[0, 4])
        best_perplexity = np.inf

        val_perplexities = []

        """
        Downloads and analyzes a bunch of random Wikipedia articles using
        online VB for LDA.
        """

        # The number of documents to analyze each iteration
        batchsize = 64
        # The total number of documents in Wikipedia
        #D = 250000
        #D = 1000
        D = 3.3e6
        # The number of topics
        K = topics_number

        # How many documents to look at
        #documentstoanalyze = int(D/batchsize)
        documentstoanalyze = 101
        
        # Our vocabulary
        #vocab = file('./dictnostops.txt').readlines()
        vocab = file('dictnostops.txt').readlines()
        W = len(vocab)

        #print 'W: ', W

        # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7

        self.olda = onlineldavb.OnlineLDA(vocab, K, D, alpha=dirichlet_alpha, eta=dirichlet_eta, tau0=1024., kappa=weight_decay, save=self.save)

        if self.save:
            self.olda.set_file_name(self.file_name)
            self.olda._rhot = self._rhot
            self.olda._lambda = self._lambda
            self.olda._Elogbeta = self._Elogbeta
            self.olda._expElogbeta = self._expElogbeta
            self.olda._updatect = self._updatect

        #print 'documentstoanalyze: ', documentstoanalyze
        # Run until we've seen D documents.

        for iteration in range(0, documentstoanalyze):
            # Download some articles
            #print 'batchsize: ', batchsize
            pre = \
                wikirandom.get_random_wikipedia_articles(batchsize)

            #print '##########################################################'
            #print pre
            #print '##########################################################'

            (docset, articlenames) = pre

            #if self.save:
            #    self.olda.set_file_name(self.file_name)
            # Give them to online LDA
            (gamma, bound) = self.olda.update_lambda_docs(docset)
            # Compute an estimate of held-out perplexity
            (wordids, wordcts) = onlineldavb.parse_doc_list(docset, self.olda._vocab)
            perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
            perplexity = numpy.exp(-perwordbound)
            val_perplexities.append(perplexity)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity

            #print 'bound: ', bound
            #print 'docset: ', len(docset)
            #print 'perwordbound: ', perwordbound
            print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
                (iteration, self.olda._rhot, perplexity)

            # Save lambda, the parameters to the variational distributions
            # over topics, and gamma, the parameters to the variational
            # distributions over topic weights for the articles analyzed in
            # the last iteration.
            if (iteration % 10 == 0):
                numpy.savetxt('lambda-%d.dat' % iteration, self.olda._lambda)
                numpy.savetxt('gamma-%d.dat' % iteration, gamma)

            return np.array([[best_perplexity]]), val_perplexities
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error