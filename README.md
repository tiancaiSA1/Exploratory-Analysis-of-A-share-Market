\documentclass[5pt,a4paper]{article}
\title{Exploratory Analysis of A-share Market}
\author
{
	\textbf{Boya Yin} 
	\\byyin@pku.edu.cn
} 
\date{December 17, 2016}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{indentfirst}
\usepackage{booktabs}
\usepackage{indentfirst}
\setlength{\parindent}{2em}
\usepackage{listings} 
\usepackage{xcolor}

\begin{document}
	\maketitle
	\begin{center}
		\tableofcontents
	\end{center}
	\newpage
	\lstset
	{
		numbers=left, 
		numberstyle= \tiny, 
		keywordstyle= \color{ blue!70},commentstyle=\color{red!50!green!50!blue!50}, 
		frame=shadowbox, 
		rulesepcolor= \color{ red!20!green!20!blue!20},
		tabsize = 2
	} 
	\section{Data Mining Introduction}
	\subsection{Data Set Introduction}
	First I want to give a first introduction of the data to make sure that I correctly understand this data set.\\
	\indent The data set consists of 100 stocks' data which have 145 features(x1-x145) and the forward yield y in some periods. The computation formula of forward yield y is $y_{t} = (prize_{t+d} - prize{t})/d$, where d is a constant represents the time difference. And the 145 features are some variables (probably we can call them indexs) . But I don't know the formula of these indexs.
	
	\subsection{Frist Thought}
	When I got the data set, the first thing I thought to do is multifactor stock selection. But after I ask senior Li, I got the message that the time difference of forward yield is only a few minuates. So I puzzled that it probably too difficult to select a basket of stocks only depend on the rise and fall of forward yield in the minuate level reasonably. But I didn't abandon the thought of stock selection, I also use a method to do this but haven't got a good result. \\
	\indent Then I think I should do some descriptive analysis before modeling. But I don't know the meaning of features, I think it is unreasonable to preprocessing the data which we called feature engineering.Of course, I have tried to use PCA , Decision Tree and some other method to select features but it seems meaningless to the final model. So I decide to use model directly without any preprocessing.\\
	\indent Although the multifactor stock selection didn't work well in minuate level, there is not influence of the prediction of tendency. Completing the stock selection and tendency prediction are what I do in this document.
	
	\subsection{Main Part}
	I have introduced the data set and first thought when I got the data. Then I will just give an outline of this document.\\
	\indent 1. \\
	\indent 2. \\
	
	\section{Multifactor Stock Selection}
	I think it is meaningless to introduce the multifactor stock selection because seniors must know much more than me about it. So I don't waste time, just start what I do. \\
	\indent I have read some papers to collect some basic idea of stock selection such as [1][2][3][4][5]. Generally they use some features which has unique meaning like NPG, ROAG, OPG... represent the growth property of a stock; PE, PB, PCF... represent the value property and so on which include fundamentals and technique index. Then the common step is to use some test method such as hypotheses testing or some probability threshold. \\
	\indent In most of the papers, after select the features index which have a significant effect on forward yield, they will develop a strategy to test how will do strategy work or some strategies to compare with each other. \\
	\indent The most common strategy is to use one or more threshold for each feature in some reasonable ways which I think is to design a decision tree (probably just a stump) in subjective way. \\
	\indent But they don't consider enough of  the interralation between features. And it is so easy and common strategy for people to build and which will result that inactivation of strategy that probably will get alpha in the last few years. So we need to take a synthetical consideration of features may influence forward yeild. Of course, machine learning is the good way to deal with this problem.
	\subsection{GBRank--Pairwise Ranking Algorithm}
	How to do sotck selection is based on what characters we want to achieve in the basket of stocks. \\
	\indent Here I just use a striaght thought that I want to select the which stocks have the highest or lowest forward yeild.\\
	\indent Under this thought, we can regard the problem as a ranking problem. There is a lot of ranking method in machine learning such as LambdaMART, GBRank, RankingSVM, RankNet ... The LambdaMART is the most famous and won the Yahoo! Learning to Rank Challenge. Here I use GBRank (Maybe I will implement LambdaMART in the future). \\
	\indent I think [6] introduced completely for GBRank algorithm, so I just write this algorithm here with some simple illustration.\\
	
	ALGORITHM (GBRank)[6]\\
	\fbox
	{
		\parbox{\textwidth}
		{
			Start with an initial guess $h_{0}$ \\
			For k = 1,2,...,K
			1. using $h_{k-1}$ as the current approximation of $h$, we separate S into two disjoint sets,:
			\[S^{+} = \{<x_{i},y_{i}> \in S | h_{k-1}(x_{i}) \geq h_{k-1}(y_{i}) + \tau\}\]
			\[S^{-} = \{<x_{i},y_{i}> \in S | h_{k-1}(x_{i}) \leq h_{k-1}(y_{i}) + \tau\}\]
			2. fitting a regression function $g_{k}(x)$ using GBDT and the following training data 
			\[\{(x_{i},h_{k-1}(y_{i})+\tau),(y_{i},h_{k-1}(x_{i})-\tau) | (x_{i},y_{i}) \in S^{-}\}\]
			3. forming (with normalization of the range of $h_{k}$)
			\[h_{k}(x) = \frac{kh_{k-1}(x)+\eta g_{k}(x)}{k+1}\]
			where $\eta$ is a shrinkage factor.\\
			\\
			Output:\quad $h_{K}$
		}
	}
	\\
	\\
	\indent It is a simple algorithm just use model to fit the set: \[\{(x_{i},h_{k-1}(y_{i})+\tau),(y_{i},h_{k-1}(x_{i})-\tau) | (x_{i},y_{i}) \in S^{-}\}\]
	and the fit model is GBDT so it called GBRank. We can see it is a pairwise solution for ranking problem.[7]\\
	\indent Then I will give an explaination of my code.
	
	\subsection{GBRank Code}
	This algorithm code consists of 4 part : 1) Preprocessing part;  2) Modelling part;  3) Practicing part;  4) Main function.
	\subsubsection{Preprocessing Part}
	

	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:24:54 2017

@author: yinboya
"""
import os
import linecache as lc
import numpy as np

class readFile:
	def __init__(self ,filepath):
		self.filepath = filepath
		self.file = []
	
	def eachFile(self):
		self.pathDir = os.listdir(self.filepath)
	
	def TDataSET(self,i):
		TSET = []
		self.eachFile()
		for filename in self.pathDir:
		#print list(eval(lc.getline(filename,i)))
		try:
		TSET.append(list(eval(lc.getline(self.filepath + filename,i))))
		except Exception:
			continue
		lc.clearcache()
		return(np.array(TSET))
	
	\end{lstlisting}
	\indent Because the data set consists of 100 *.csv files. This part just read all of them for specify rows. Pay attention to the computer memories, because this part use package "linecache" which can read lines very fast but will occupy a lot of memories.(of course the code can be changed into low memory cost such as just using "open")\\
	
	\subsubsection{Modelling Part}
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:55:30 2017

@author: yinboya
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:55:30 2017

@author: yinboya
"""

'''
gbrank
'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
#from utils import choose_threshold
#import itertools
import numpy as np



class GBRank(BaseEstimator, ClassifierMixin):
	""" rank
	# Start with an initial guess h0, for k = 1, 2, ...
	1) using h_{k-1} as the current approximation of h, we separate S into two
	disjoint sets, S+ = {(x_i, y_i) e S | h_{k-1}(x_i) >= h_{k-1}(y_i)+tau} and
	S- = {(x_i, y_i) e S | h_{k-1}(x_i) < h_{k-1}(y_i)+tau}
	2) fitting a regression function g_k(x) using GBT and the following training
	data {(x_i,h_{k-1}(yi)+tau), (y_i,h_{k-1}(x_i)-tau) | (xi, yi) e S-}
	3) forming (with normalization of the range of h_k) h_k(x) =
	k h_{k-1}(x)+eta g_k(x), k+1 where eta is a shrinkage factor
	
	Returns
	-------
	self
	"""
	def __init__(self, base_estimator=None, max_its= 100, eta = 0.05, \
	s_rate = 0.9):
		super(GBRank, self).__init__()
		self.g = [] # the set of classifierMixin
		self.eta = eta # 0.05
		self.kmax = max_its # maximum iter nums
		self.s_rate = s_rate 
		# iteration will stop when accuracy rate exceed the s_rate
		self.train_s_rate = []
		if base_estimator is None:
			base_estimator = GradientBoostingRegressor()
		self.estimator = base_estimator
		
	# Estranhamente, este código até parece mais rápido que o anterior!
	
	def fit(self, X, y, tau = 1):
		#tau = 1  # [0,1]
		Xs = []
		H = []
		for i in xrange(len(X)):
			for j in xrange(len(X)):
				if y[i] > y[j]:
					Xs.append(X[i])  # y[i] > y[i+1]
					Xs.append(X[j])
					H.append(2)
					H.append(0)
		l = np.ones(len(H), bool)
		Xs = np.asarray(Xs)  # as matrix
		H = np.asarray(H).T  # as matrix
		NXs = Xs.shape[0]
		while self.kmax > len(self.g):
			g = clone(self.estimator).fit(Xs[l], H[l])
			self.g.append(g)
			l = np.ones(len(H), bool)
			H = self.predict_proba(Xs).T  # as matrix
			
			for i in xrange(len(H)-2, -1, -2):
				assert (i % 2) == 0
					if H[i] >= H[i+1]+tau:
					l[i] = False
					l[i+1] = False
				else:
					Hi = H[i+1]+tau
					H[i+1] = H[i]-tau
					H[i] = Hi
			#if np.sum(l) == 0:  # converged
			#    break
			if np.sum(l) <= NXs * (1-self.s_rate):
				break
		self.train_s_rate.append(1 - np.sum(l) / NXs)
		#H = self.predict_proba(X)
		#self.th = choose_threshold(H, y)
		return self
	
	def update_fit(self, X, y, tau = 1):
		Xs = []
		for i in xrange(len(X)):
			for j in xrange(len(X)):
				if y[i] > y[j]:
					Xs.append(X[i])  # y[i] > y[i+1]
					Xs.append(X[j])
		l = np.ones(len(Xs), bool)
		Xs = np.asarray(Xs)  # as matrix
		NXs = Xs.shape[0]
		while self.kmax > len(self.g):
			H = self.predict_proba(Xs).T  # as matrix
			for i in xrange(len(H)-2, -1, -2):
				assert (i % 2) == 0
				if H[i] >= H[i+1]+tau:
					l[i] = False
					l[i+1] = False
				else:
					Hi = H[i+1]+tau
					H[i+1] = H[i]-tau
					H[i] = Hi
			if np.sum(l) <= NXs * (1-self.s_rate):
				break
		
			g = clone(self.estimator).fit(Xs[l], H[l])
			self.g.append(g)
			l = np.ones(len(H), bool)
		self.train_s_rate.append(1 - np.sum(l) / NXs)
		return self
	
	
	
	def hk(self, X, k):
		if k == 0:
			s = self.g[0].predict(X)
			return s
		g = self.g[k].predict(X)
		h = self.hk(X, k-1)
		return (k*h + self.eta*g) / float(k+1)
	
	def predict_proba(self, X):
		# these are not probabilities, but I overloaded this function because
		# it makes it nicer to have a common interface for the ROC curves
		X = np.asarray(X)
		return self.hk(X, len(self.g)-1)
	
	def predict(self, X):
		return self.predict_proba(X)
		# return (self.predict_proba(X) >= self.th).astype(int)

	\end{lstlisting}
	\indent This part mainly defined the algorithm which we illustrated in 2.1. And there is some different that I used whole data to train the model as $h_{0}$ at first. And inorder to make full use of the data, I defined an update\_fit function which using $h_{K}$ that trained by fit function as the new $h_{0}$ for new data set (You can thought this as a priori information). 
	\subsubsection{Practicing Part}
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:32:48 2017

@author: yinboya
"""

import trainAndtest_SET as TT
import rankmodel as rmo
import numpy as np

class testGBRank:
	'''
	init as train,test set index of the whole data
	input the list of train,test index
	'''
	def __init__(self,trainInd,testInd,filepath):
		self.trainI = trainInd
		self.testI = testInd
		self.filepath = filepath
		self.correctRate = []
		self.DataFP = TT.readFile(self.filepath)
		#self.trainNum = len(trainInd) # number of train set
		#self.testNum = len(testInd) # number of test set
	
	def trainRank(self):
		self.rmodel = rmo.GBRank()
		X = self.DataFP.TDataSET(self.trainI[0])
		y = X[-1]
		X = X[:-1]
		self.rmodel.fit(X,y)
		self.trainI = self.trainI[1:]
		for Ind in self.trainI:
			X = self.DataFP.TDataSET(Ind)
			y = X[-1]
			X = X[:-1]
			self.rmodel.update_fit(X,y)
	
	def testRank(self):
		if len(self.testI) == 1:
		Ind = self.testI
		X = self.DataFP.TDataSET(Ind)
		y = X[-1]
		X = X[:-1]
		Xs = []
		for i in xrange(X.shape[0]):
			for j in xrange(X.shape[0]):
				if y[i] > y[j]:
				Xs.append(X[i])  # y[i] > y[i+1]
				Xs.append(X[j])
		l = np.ones(len(Xs), bool)
		Xs = np.asarray(Xs)  # as matrix
		H = self.rmodel.predict_proba(Xs).T  # as matrix
		for i in xrange(len(H)-2, -1, -2):
			assert (i % 2) == 0
			if H[i] >= H[i+1]:
				l[i] = False
				l[i+1] = False
		self.correctRate.append(1 - np.sum(l) / float(len(l)))
		return 0
	
	for Ind in self.testI:
		X = self.DataFP.TDataSET(Ind)
		y = X[-1]
		X = X[:-1]
		Xs = []
		for i in xrange(X.shape[0]):
			for j in xrange(X.shape[0]):
				if y[i] > y[j]:
					Xs.append(X[i])  # y[i] > y[i+1]
					Xs.append(X[j])
		l = np.ones(len(Xs), bool)
		Xs = np.asarray(Xs)  # as matrix
		H = self.rmodel.predict_proba(Xs).T  # as matrix
		for i in xrange(len(H)-2, -1, -2):
			assert (i % 2) == 0
			if H[i] >= H[i+1]:
				l[i] = False
				l[i+1] = False
		self.correctRate.append(1 - np.sum(l) / float(len(l)))
	
	def test_Correct_Rate(self):
		return self.correctRate
	
	def hk(self, X):
		return(self.rmodel.predict_proba(X))
	
	def sortHK(self, X):
		hk_value = self.hk(X)
		return(np.argsort(-hk_value))

	
	\end{lstlisting}
	\indent In order to combine the 2.2.1 and 2.2.2, I defined the testGBRank to make more simple for training model and test it.
	\subsubsection{Main Function}
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
	import testRank as tR
	import numpy as np
	
	trainInd = np.array(range(2)) + 2
	
	testInd = np.array(range(5)) + 3
	
	filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
	testGBRank = tR.testGBRank(trainInd,testInd,filepath)
	testGBRank.trainRank()
	testGBRank.testRank()
	testGBRank.clearlc()
	testGBRank.test_Correct_Rate()
	\end{lstlisting}
	\indent You can just use command "python main.py" to start function. You only need to set the lines of training data set(such as "np.array(range(2)) + 2" means line2,line3 are the training set of the model) and the lines of testing data set. Then call the test\_Correct\_Rate() function of package testRank which will show the accurancy rate of the GBRank algorithm in testing data.
	\subsection{Case}
	Calling the main function we can get the accurancy rate which define by pairwise judgement.
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GBRank}
		\centering{\caption{Testing of GBRank}}
		\label{fig:D+B}
	\end{figure}
	\\
	\section{Tendency Forecast}
	This Part I mainly focus on the parameters selection for different model. Make a case like 2.3 for a model is easy. But decide the best parameters (or super parameters).It probably can be easy in some simple model which has 1~3 parameters like support vector mechine, but it also can be a difficult problem especially in some sophsiticated model like gbdt,xgboost which have 5~10 even more parameters for us to decide.\\
	\indent Probably for some experienced people we will use common sense to judge a range for some paramters like max\_depth:6~8 for xgboost is usually a good range in cross validation performation. But it still a subjective way to select parameters value.\\
	\indent In this document, I use Genetic Algorithm to help me seach the effective parameters (probably not best but superior)\\
	
	\subsection{GA -- Support Vector Mechine}
	\subsubsection{Genetic Algorithm}
	I think I don't need to give an introduction to GA, but probably I have to illustrate the population(individual), evalution, crossover, mate, mutation and selection method in GA.\\
	I used deap package which is a opensource python project to achieve GA. If you want to understand my code, probably you should read the tutorial of deap[8]
	\subsubsection{SVM -- evaluation}
	SVM probably is a very simple model which only have 2 parameters which are important for us to select : 1. kernel funtion; 2. C (we can call it regularization).\\
	\indent Because its simple, a lot of researcher just use grid search (I think it is a exhaustive search which will cost a lot of time) to find good parameter. So it can be a good example.\\
	\indent First we need to decide the objective function. GA required that to design a evalution function to evaluate the each individual, so that it could iterate by select function(we call it natural selection or survival of the fittest). \\
	I define the evalution by mean of the cross validation MSE. The function code is
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
def svmc_evaluation(individual):
	N_SPLITS = 2
	kf = KFold(n_splits = N_SPLITS)
	cv_mseValue = 0
	Kernel = ['linear','rbf','poly','sigmoid']
	fc = svm.SVR(C = individual[0], kernel = Kernel[individual[1]])
	for train, test in kf.split(trainX):
		fc.fit(trainX[train,:], trainY[train])
		cv_mseValue += sum((trainY[test]
	fc.predict(trainX[test,:])) ** 2) / len(test)
	cv_mseValue = cv_mseValue / N_SPLITS
	return(cv_mseValue,)
	\end{lstlisting}
	The code select the split number such as 2 folds to do cross validation and calculate the MSE of the test data set. So the objective function is cross validation mean MSE.
	
	\subsubsection{SVM -- mutation}
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
	def svmc_mutation(individual):
		delta = 10 # the delta of Guassian
		if random.random() < 0.5:
			tmp = random.gauss(individual[0],delta)
			while tmp < 0:
				tmp = random.gauss(individual[0],delta)
			individual[0] = tmp
		else:
			individual[1] = random.randint(0,3)
	\end{lstlisting}
	\indent I think it is simple enough that I can't illustrate it well.
	
	\subsubsection{SVM -- algorithm}
	The select function and crossover function are tools.selTournament and tools.cxTwoPoint, which the meaning can see in [8].\\
	The algorithm is final step:
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
def main(NGEN = 10,CXPB = 0.5, MUTPB = 0.2):
	# Creating the Population
	pop = toolbox.population(n=5)
	# Evaluate the entire population
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	# Begin the evolution
	for g in range(NGEN):
		print("-- Generation %i --" % g)
		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values
			
		for mutant in offspring:
			if random.random() < MUTPB:
			toolbox.mutate(mutant)
			del mutant.fitness.values
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		pop[:] = offspring
		
		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]
		
		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5
		
		print("  Min %s" % min(fits))
		print("  Max %s" % max(fits))
		print("  Avg %s" % mean)
		print("  Std %s" % std)
		print("  Parameter %s" % offspring)
	\end{lstlisting}
	\subsubsection{case}
	Then I use sym\_1.csv to test this method, the result is figure2 and figure3.
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GAsvm1}
		\centering{\caption{The parameters selection and mse of GA--SVM Generation 0}}
		\label{fig:D+B}
	\end{figure}
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GAsvm2}
		\centering{\caption{The parameters selection and mse of GA--SVM Generation 10}}
		\label{fig:D+B}
	\end{figure}
	\subsection{GA -- xgboost}
	Although the xgboost have a lot more parameters than SVM but the core thoughts is just same.\\
	\subsubsection{xgboost -- code}
	So I won't illustrate it, just give the code:
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 03:28:04 2017

@author: yinboya
"""


from __future__ import division
import numpy as np
from sklearn.model_selection import KFold
import GenDataSet

I = 1
'''
represent the(classifcation of regression) model:
1: classfication
2: regression
'''
popNum = 10 # Population
N_splits = 2 # Crossvalidation fold number
iter_NGEN = 10 # Generation number
iter_CXPB = 0.8 # Crossover probability
iter_MUTPB = 0.9 # Mutation probability 

classMaxThreshold = 0.001
classMinThreshold = -0.001

# Read Data
if I == 1:
	filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
	name = 'sym_1.csv'
	myrf = GenDataSet.csv2tset(filepath = filepath, name = name)
	trainNum = 1000
	testNum = 500
	trainX, trainY = myrf.tData_Classification(trainNum,
	Class_MaxThreshold = classMaxThreshold,
	Class_MinThreshold = classMinThreshold)
	testX, testY = myrf.tData_Classification(testNum,
	Class_MaxThreshold = classMaxThreshold,
	Class_MinThreshold = classMinThreshold)
else:
	filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
	name = 'sym_1.csv'
	myrf = GenDataSet.csv2tset(filepath = filepath, name = name)
	trainNum = 1000
	testNum = 500
	trainX, trainY = myrf.tData_Regression(trainNum)
	testX, testY = myrf.tData_Regression(testNum)

# Setting Things Up
import random
from deap import base
from deap import creator
from deap import tools

# Creator
if I == 1:
	creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0,1.0))
	creator.create("Individual", list, fitness=creator.FitnessMulti)
else:
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)
	
toolbox = base.Toolbox()

'''
Attribute generator 
'''
"""
General Parameters:
booster[default = gbtree]
silent[default = 0]
nthread[default to maximum number of threads available if not set]
"""

"""
Booster Parameters:
eta[default = 0.3]
min_child_weight[default = 1]:
define the minimum sum of weight of all observations required in a child
This is similar to min_child_leaf in GBM but not exactly. This refers to 
min “sum of weights” of observations while GBM has min “number of observations”.
Used to control over-fitting. Higher values prevent a model from learning 
relations which might be highly specific to the particular sample selected 
for a tree.
Too high values can lead to under-fitting hence, it should be tuned using CV.
max_depth[default = 6]:
typical:3-10
max_leaf_nodes:
If this is defined, GBM will ignore max_depth.
gamma[default = 0]:
A node is split only when the resulting split gives a positive reduction in 
the loss function. Gamma specifies the minimum loss reduction required to 
make a split.
Makes the algorithm conservative. The values can vary depending on the loss
function and should be tuned.
#max_delta_step[default=0]:
In maximum delta step we allow each tree’s weight estimation to be.
This is generally not used but you can explore further if you wish.
subsample [default=1]:
Same as the subsample of GBM. Denotes the fraction of observations 
to be randomly samples for each tree.
Typical values: 0.5-1
colsample_bytree[default=1]:
Similar to max_features in GBM. Denotes the fraction of columns to be 
randomly samples for each tree.
Typical values: 0.5-1
#colsample_bylevel [default=1]
#lambda [default=1]:
L2 regularization term on weights (analogous to Ridge regression)
This used to handle the regularization part of XGBoost. Though many 
data scientists don’t use it often, it should be explored to reduce overfitting.
#alpha [default=0]:
L1 regularization term on weight (analogous to Lasso regression)
Can be used in case of very high dimensionality so that the algorithm runs 
faster when implemented
#scale_pos_weight [default=1]:
A value greater than 0 should be used in case of high class imbalance as it 
helps in faster convergence.
"""
def xgb_eta(MAXeta = 0.2, MINeta = 0.01):
	return((MAXeta - MINeta)*random.random() + MINeta)

def xgb_MinChildWeight(MAXmcw = 20, MINmcw = 5):
	return(random.randint(MINmcw,MAXmcw))

def xgb_maxdepth(MAXmd = 13, MINmd = 3):
	return(random.randint(MINmd,MAXmd))

def xgb_subsample(MAXss = 1, MINss = 0.4):
	return((MAXss - MINss)*random.random() + MINss)

def xgb_colsample(MAXcs = 1, MINcs = 0.4):
	return((MAXcs - MINcs)*random.random() + MINcs)

# register xgb parameters
toolbox.register("attr_xgb_eta", xgb_eta)
toolbox.register("attr_xgb_MinChildWeight", xgb_MinChildWeight)
toolbox.register("attr_xgb_maxdepth", xgb_maxdepth)
toolbox.register("attr_xgb_subsample", xgb_subsample)
toolbox.register("attr_xgb_colsample", xgb_colsample)


"""
Learning Task Parameters:
#objective [default=reg:linear]:
This defines the loss function to be minimized. Mostly used values are:
binary:logistic –logistic regression for binary classification, returns p
redicted probability (not class)

multi:softmax –multiclass classification using the softmax objective, 
returns predicted class (not probabilities):
you also need to set an additional num_class (number of classes) 
parameter defining the number of unique classes

multi:softprob –same as softmax, but returns predicted probability 
of each data point belonging to each class.
#eval_metric [ default according to objective ]:
The metric to be used for validation data.
The default values are rmse for regression and error for classification.
Typical values are:
rmse – root mean square error
mae – mean absolute error
logloss – negative log-likelihood
error – Binary classification error rate (0.5 threshold)
merror – Multiclass classification error rate
mlogloss – Multiclass logloss
auc: Area under the curve
#seed [default=0]:
"""
"""
If you’ve been using Scikit-Learn till now, 
these parameter names might not look familiar. 
A good news is that xgboost module in python has an 
sklearn wrapper called XGBClassifier. It uses sklearn 
style naming convention. The parameters names which will change are:

eta –> learning_rate
lambda –> reg_lambda
alpha –> reg_alpha
"""


# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
(toolbox.attr_xgb_eta,toolbox.attr_xgb_MinChildWeight,
toolbox.attr_xgb_maxdepth,toolbox.attr_xgb_subsample,
toolbox.attr_xgb_colsample), 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The Evaluation Function
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

def XGB_class_evaluation(individual):
	N_SPLITS = N_splits
	kf = KFold(n_splits = N_SPLITS)
	fc = XGBClassifier(learning_rate = individual[0], n_estimators = 100,
	silent = True,
	nthread = -1, gamma = 0,
	min_child_weight = individual[1],max_depth = individual[2],
	subsample = individual[3],colsample_bylevel = individual[4],
	seed = 0)
	M_pos = 0
	M_mid = 0
	M_neg = 0
	for train, test in kf.split(trainX):
	fc.fit(trainX[train,:], trainY[train])
	testY_pre = fc.predict(trainX[test,:])
	Ind_pos = (trainY[test] == 1)
	Ind_mid = (trainY[test] == 0)
	Ind_neg = (trainY[test] == -1)
	M_pos += len(np.where(np.array(testY_pre[Ind_pos]) == 1)[0]) / len(np.where(Ind_pos)[0])
	M_mid += len(np.where(np.array(testY_pre[Ind_mid]) == 0)[0]) / len(np.where(Ind_mid)[0])
	M_neg += len(np.where(np.array(testY_pre[Ind_neg]) == -1)[0]) / len(np.where(Ind_neg)[0])
	
	correct = map(lambda x : x / N_SPLITS, [M_pos, M_mid, M_neg])
	return(tuple(correct))

def XGB_reg_evaluation(individual):
	N_SPLITS = N_splits
	kf = KFold(n_splits = N_SPLITS)
	cv_mseValue = 0
	fc = XGBRegressor(learning_rate = individual[0], n_estimators = 100,
	silent = True, objective = "reg:linear",
	nthread = -1, gamma = 0,
	min_child_weight = individual[1],max_depth = individual[2],
	subsample = individual[3],colsample_bylevel = individual[4],
	seed = 0)
	for train, test in kf.split(trainX):
	fc.fit(trainX[train,:], trainY[train])
	cv_mseValue += sum((trainY[test] - fc.predict(trainX[test,:])) ** 2) / len(test)
	cv_mseValue = cv_mseValue / N_SPLITS
	return(cv_mseValue,)

# The mutate Function
def xgb_mutation(individual):
	N = len(individual)
	rn = random.randint(1,N)
	if rn == 1 :
		individual[0] = toolbox.attr_xgb_eta()
	elif rn == 2 :
		individual[1] = toolbox.attr_xgb_MinChildWeight()
	elif rn == 3 :
		individual[2] = toolbox.attr_xgb_maxdepth()
	elif rn == 4 :
		individual[3] = toolbox.attr_xgb_subsample()
	elif rn == 5 :
		individual[4] = toolbox.attr_xgb_colsample()


# The Genetic Operators
if I == 1:
	toolbox.register("evaluate", XGB_class_evaluation)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", xgb_mutation)
	toolbox.register("select", tools.selTournament, tournsize = 3)
	def main(NGEN = iter_NGEN, CXPB = iter_CXPB, MUTPB = iter_MUTPB):
		# Creating the Population
		pop = toolbox.population(n=popNum)
		# Evaluate the entire population
		fitnesses = list(map(toolbox.evaluate, pop))
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit
		# Begin the evolution
		for g in range(NGEN):
			print("-- Generation %i --" % g)
			# Select the next generation individuals
			offspring = toolbox.select(pop, len(pop))
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))
			# Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < CXPB:
					toolbox.mate(child1, child2)
					del child1.fitness.values
					del child2.fitness.values
				
			for mutant in offspring:
				if random.random() < MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values
			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			pop[:] = offspring
			
			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values for ind in pop]
			fits = np.array(fits)
			
			mean = np.mean(fits, axis = 0)
			std = np.std(fit, axis = 0)
			
			print("  Min %s" % np.min(fits,axis = 0))
			print("  Max %s" % np.max(fits,axis = 0))
			print("  Avg %s" % mean)
			print("  Std %s" % std)
			print("  Parameter %s" % tools.selBest(offspring,1))

else:
	toolbox.register("evaluate", XGB_reg_evaluation)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", xgb_mutation)
	toolbox.register("select", tools.selTournament, tournsize = 3)
	def main(NGEN = iter_NGEN, CXPB = iter_CXPB, MUTPB = iter_MUTPB):
		# Creating the Population
		pop = toolbox.population(n=popNum)
		# Evaluate the entire population
		fitnesses = list(map(toolbox.evaluate, pop))
		for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
		# Begin the evolution
		for g in range(NGEN):
			print("-- Generation %i --" % g)
			# Select the next generation individuals
			offspring = toolbox.select(pop, len(pop))
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))
			# Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < CXPB:
					toolbox.mate(child1, child2)
					del child1.fitness.values
					del child2.fitness.values
			
			for mutant in offspring:
				if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values
			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			pop[:] = offspring
			
			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values[0] for ind in pop]
			
			length = len(pop)
			mean = sum(fits) / length
			sum2 = sum(x*x for x in fits)
			std = abs(sum2 / length - mean**2)**0.5
			
			print("  Min %s" % min(fits))
			print("  Max %s" % max(fits))
			print("  Avg %s" % mean)
			print("  Std %s" % std)
			print("  Parameter %s" % offspring[np.argmin(fits)])



if __name__ == '__main__':
	main()
	
	\end{lstlisting}
	\subsubsection{xgboost -- readFile}
	\begin{lstlisting}[language = python,numbers=left, numberstyle=\tiny,keywordstyle=\color{blue!70},commentstyle=\color{red!50!green!50!blue!50},frame=shadowbox, rulesepcolor=\color{red!20!green!20!blue!20}]
	#!/usr/bin/env python2
	# -*- coding: utf-8 -*-
	"""
	Created on Thu Jan 26 20:24:54 2017
	
	@author: yinboya
	"""
	
	import numpy as np
	import pandas as pd
	
	class csv2tset:
		def __init__(self, filepath, name):
			self.filepath = filepath + name
			self.Data = pd.read_csv(self.filepath, iterator = True)
			
			def tData_Regression(self, tNum):
			tData = self.Data.get_chunk(tNum)
			tData = np.array(tData)
			tX, tY = tData[:,:-1], tData[:,-1] * 100
			return(tX,tY)
		
		def tData_Classification(self, tNum, Class_MaxThreshold, Class_MinThreshold):
			tData = self.Data.get_chunk(tNum)
			tData = np.array(tData)
			tX, tY = tData[:,:-1], tData[:,-1]
			TMaxInd = []
			TMinInd = []
			TMidInd = []
			for i in range(len(tY)):
				if tY[i] > Class_MaxThreshold:
					TMaxInd.append(i)
				elif tY[i] < Class_MinThreshold:
					TMinInd.append(i)
				else:
					TMidInd.append(i)
			tY[TMaxInd] = 1
			tY[TMinInd] = -1
			tY[TMidInd] = 0
			return(tX,tY)
	\end{lstlisting}
	
	\subsubsection{case1 -- Regression}
	Please make sure that when you use "python GAxgboost.py", the GenDataSet is in the current directory. And you should specify the data filepath such as (~/Document/Data/sym\_1.csv)\\
	\indent The result of regression can compare with 2.3 like figure4 and figure5
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GAxgboostReg1}
		\centering{\caption{The xgboost regression of tendency forecast -- MSE generation0}}
		\label{fig:D+B}
	\end{figure}
	
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GAxgboostReg2}
		\centering{\caption{The xgboost regression of tendency forecast -- MSE generation100}}
		\label{fig:D+B}
	\end{figure}
	
	\subsubsection{case2 -- Classfication}
	I also code the classfication problem because we usually need to decide when to establish long or short position. So I divided the data set into 3 class by this rule:
	\indent Decide threshold $thr_{+1}$ and $thr_{-1}$.\\
	\indent 1. $y_{t} > thr_{+1}$ , class +1\\
	\indent 2. $y_{t} < thr_{-1}$ , class -1\\
	\indent 3. $else$ , class 0\\
	Then change the variable I into 1 in the GAxgboost.py like figure 6.
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GAxgboostChange}
		\centering{\caption{The xgboost regression of tendency forecast -- MSE generation0}}
		\label{fig:D+B}
	\end{figure}
	 And just use "python GAxgboost.py" the model will run well like figure7 and figure8.
	 \begin{figure}[htbp]
	 	\centering
	 	\includegraphics[width=4.00in,height=2.50in]{GAxgboostClass1}
	 	\centering{\caption{The xgboost regression of tendency forecast -- MSE generation0}}
	 	\label{fig:D+B}
	 \end{figure}
	 
	\begin{figure}[htbp]
		\centering
		\includegraphics[width=4.00in,height=2.50in]{GAxgboostClass2}
		\centering{\caption{The xgboost regression of tendency forecast -- MSE generation5}}
		\label{fig:D+B}
	\end{figure}

	\section{Conclusion}
	I will give a conclusion of what I did in this document and why. There is not enough message for me to make a trade strategy. The main idea of this document is just to test the model and try to mining something may be usable or meaningful. \\
	\indent Why I didn't preprocess the data ? First of all, to the features, I don't know the meanings of them, it is unreasonable to take feature engineering if we know nothing about the characters of features. Then I can't use time series model because the data is extracted by random sampling, which is why it is impossible to make arma,garch model to get a stationary process or other things. So I just use this data to test the model and try to do stocks selection and tendency forecast which I think are two of the most important things when we analyse the stock market.\\
	\indent I know I didn't illustrate my work completely in this document, if there are some doubt about it, please Phone,Email or just WeChat me.
	
	\bibliographystyle{unsrt}
	\begin{thebibliography}{99}
		\bibitem{wu:chfont}
		Hu Yanyan, The significance of the spread structure to the futures price discovery and hedging function -- Based on the analysis of the international crude oil and domestic fuel oil futures market,  Fudan University Master's thesis, 27/3/2017
		\bibitem{wu:chfont}
		Liu Yi, An empirical study on the stock market in China factor model, Fudan University Master's thesis, 3/2012
		\bibitem{wu:chfont}
		Li Boya, An empirical study on momentum alpha investment strategy based on A stock market, East China Normal University Master's Thesis, 5/2016 
		\bibitem{wu:chfont}
		Hu Qian ,Study on quantitative stock selection based on machine learning, 25/5/2016
		\bibitem{wu:chfont}
		Zhang Hongfan, The theory and practice of using quantitative investment strategy to realize Alpha, Zhejiang University, Master's Thesis, 5/2015
		\bibitem{wu:chfont}
		Zhaohui Zheng，Hongyuan Zha, Keke Chen, Gordon Sun, A Regression Framework for Learning Ranking Functions Using Relative Relevance Judgments, Sigir: International Acm Sigir Conference on Research \& Development in Information Retrieval,2007:287-294
		\bibitem{wu:chfont}
		https://en.wikipedia.org/wiki/Learning\_to\_rank
		\bibitem{wu:chfont}
		http://deap.gel.ulaval.ca/doc/default/index.html
	\end{thebibliography}
\end{document}
