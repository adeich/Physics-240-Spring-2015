from sklearn import datasets
from sklearn import svm


def make_data_array_for_ML(list_of_sample_dicts):

	all_data_list = []
	all_target_list = []

	for sample_dict in list_of_sample_dicts:
		all_data_list.append(sample_dict['data'])
		all_target_list.append(sample_dict['target'])

	return np.array(all_data_list), np.array(all_target_list)


def simple_test():

	# SVC is Support Vector Classification.
	digits = datasets.load_digits()
	classification = svm.SVC(gamma=0.001, C=100.)
	classification.fit(digits.data[:-1], digits.target[:-1])
	print	classification.predict(digits.data[-1])




if __name__ == '__main__':
	simple_test()
