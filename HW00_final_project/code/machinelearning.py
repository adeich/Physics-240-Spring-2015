import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import grid_search
import pickle


def make_data_array_for_ML(list_of_sample_dicts):

	all_data_list = []
	all_target_list = []

	for sample_dict in list_of_sample_dicts:
		all_data_list.append(sample_dict['data'])
		all_target_list.append(sample_dict['target'])

	return np.array(all_data_list), np.array(all_target_list)

def compile_classification(pickle_filename, data_array, target_array):
	
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svr = svm.SVC()
	classification = grid_search.GridSearchCV(svr, parameters)
	print(classification.fit(data_array, target_array))
	with open(pickle_filename, 'w') as f:
		pickle.dump(classification, f)
	print('Machine learning model written to {}'.format(pickle_filename))	


def simple_classification(data_array, target_array, vector_in_question):

	classification = svm.SVC(gamma=0.001, C=100)
	classification.fit(data_array, target_array)
	print(classification.predict(vector_in_question))


def classify_from_pickle(pickled_filepath, normalized_spectrum_path):
	with open(pickled_filepath, 'r') as p:
		classification = pickle.load(p)
	with open(normalized_spectrum_path, 'r') as test_item:
		return classification.predict(np.loadtxt(test_item))


def simple_test():

	# SVC is Support Vector Classification.
	digits = datasets.load_digits()
	classification = svm.SVC(gamma=0.001, C=100.)
	classification.fit(digits.data[:-1], digits.target[:-1])
	print	classification.predict(digits.data[-1])




if __name__ == '__main__':
	simple_test()
