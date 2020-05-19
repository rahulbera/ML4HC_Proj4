import time
import numpy as np
import pandas as pd
import shogun as sg

def saveInfo(name, my_time, my_acc, my_F1, my_precision, my_recall, timeDict, accDict, F1Dict, precisionDict, recallDict):
	timeDict[name] = my_time
	accDict[name] = my_acc
	F1Dict[name]= my_F1
	precisionDict[name] = my_precision
	recallDict[name] = my_recall
	# print("---{}: {} seconds ---".format (name, my_time))
	# print('{}, {} accuracy score: {}, F1-score: {}'.format(title, name, accuracy_score, my_F1))

def evaluateAndSave(name, train_time, features_test, labels_test, labels_predict, timeDict, accDict, F1Dict, precisionDict, recallDict):
	acc = sg.AccuracyMeasure().evaluate(labels_predict, labels_test)
	f1 = sg.F1Measure().evaluate(labels_predict, labels_test)
	prec = sg.PrecisionMeasure().evaluate(labels_predict, labels_test)
	rec = sg.RecallMeasure().evaluate(labels_predict, labels_test)
	print('===========================================')
	print('Model: ', name)
	print('Train Time: ', train_time)
	print('Accuracy:', acc)
	print('F1:', f1)
	print('Precision:', prec)
	print('Recall:', rec)
	saveInfo(name, train_time, acc, f1, prec, rec, timeDict, accDict, F1Dict, precisionDict, recallDict)
	print('===========================================\n')


train_data_source = './exercise_data/human_dna_train_split_5_95.csv'
test_data_source = './exercise_data/human_dna_test_split.csv'

train_df = pd.read_csv(train_data_source, header=0)
test_df = pd.read_csv(test_data_source, header=0)

train_seq = train_df['sequences']
train_label = train_df['labels']
test_seq = test_df['sequences']
test_label = test_df['labels']

train_seq = np.array(train_seq)
train_label = np.array(train_label)
test_seq = np.array(test_seq)
test_label = np.array(test_label)

print('train_seq shape:', train_seq.shape)
print('test_seq shape:', test_seq.shape)

timeDict = {}
accDict = {}
F1Dict = {}
precisionDict = {}
recallDict = {}

features_train = sg.StringCharFeatures(train_seq.tolist(), sg.DNA)
features_test = sg.StringCharFeatures(test_seq.tolist(), sg.DNA)
labels_train = sg.BinaryLabels(train_label)
labels_test = sg.BinaryLabels(test_label)

# SVM
C = 1.0
epsilon = 0.001

svm = sg.LibSVM(C, sg.WeightedDegreeStringKernel(features_train, features_train, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("WeightedDegreeStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

svm = sg.LibSVM(C, sg.WeightedDegreePositionStringKernel(features_train, features_train, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("WeightedDegreePositionStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

# Does not work
'''
svm = sg.LibSVM(C, sg.PolyMatchStringKernel(features_train, features_train, 5, True), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("PolyMatchStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''

# Does not work
'''
svm = sg.LibSVM(C, sg.FixedDegreeStringKernel(features_train, features_train, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("FixedDegreeStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''

svm = sg.LibSVM(C, sg.LinearStringKernel(features_train, features_train), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("LinearStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

# Takes too much time
'''
svm = sg.LibSVM(C, sg.LocalAlignmentStringKernel(features_train, features_train), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("LocalAlignmentStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''

# Does not work
'''
svm = sg.LibSVM(C, sg.LocalityImprovedStringKernel(features_train, features_train, 10, 5, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("LocalityImprovedStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''

svm = sg.LibSVM(C, sg.GaussianMatchStringKernel(features_train, features_train, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("GaussianMatchStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

svm = sg.LibSVM(C, sg.CommUlongStringKernel(features_train, features_train), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("CommUlongStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

'''
svm = sg.LibSVM(C, sg.CommWordStringKernel(features_train, features_train), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("CommWordStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

svm = sg.LibSVM(C, sg.MatchWordStringKernel(features_train, features_train, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("MatchWordStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''

# Does not work
'''
svm = sg.LibSVM(C, sg.SimpleLocalityImprovedStringKernel(features_train, features_train, 10, 5, 5), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("SimpleLocalityImprovedStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''
'''
svm = sg.LibSVM(C, sg.SNPStringKernel(features_train, features_train, 5, 10, True), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("SNPStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''
'''
svm = sg.LibSVM(C, sg.WeightedCommWordStringKernel(features_train, features_train), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("WeightedCommWordStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''
'''
svm = sg.LibSVM(C, sg.SparseSpatialSampleStringKernel(features_train, features_train), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("SparseSpatialSampleStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)
'''

svm = sg.LibSVM(C, sg.OligoStringKernel(features_train, features_train, 10, 8), labels_train)
svm.set_epsilon(epsilon)
start = time.time()
svm.train()
evaluateAndSave("OligoStringKernel", time.time()-start, 
				features_test, labels_test, svm.apply_binary(features_test),
				timeDict, accDict, F1Dict, precisionDict, recallDict)

