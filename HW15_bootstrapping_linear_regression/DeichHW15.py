import numpy as np
import matplotlib.pyplot as plt

def make_poly(a,x) : # polynomial model of vector x with parameters a
	#print 'a:{}'.format(a)
	M = len(a)-1
	y_mod = np.zeros(len(x)) 
	for j in range(M+1) :
		y_mod += a[j] * x**j 
	return y_mod
 	

# Find parameters for linear regression of data.
def calculate_fit_params(x_array, y_array, sigma_array) :

	S = np.sum(1/sigma_array**2)
	Sx = np.sum(x_array/sigma_array**2)
	Sy = np.sum(y_array/sigma_array**2)
	Sxx = np.sum((x_array/sigma_array)**2)
	Sxy = np.sum(x_array*y_array/sigma_array**2)
	S_delta = Sxx*S - Sx**2

	intercept = (Sxx*Sy - Sxy*Sx) / S_delta
	slope = (Sxy*S - Sx*Sy) / S_delta

	intercept_error = np.sqrt(Sxx / S_delta)
	slope_error = np.sqrt(S / S_delta)

	y_fit = slope * x_array + intercept

	chi_squared = np.sum((y_fit-y_array)**2/sigma_array**2) / (len(x_array)-2)

	return {'slope': slope, 'intercept': intercept, 
		'slope_error': slope_error, 'intercept_error': intercept_error,
		'chi_squared': chi_squared,
		'poly_params': [intercept, slope]}

# custom function for preparing temperature data from file. Adds faked uncertainty.
def take_data_make_x_y_sigma(data_array, add_gaussian_error=False):
	x_column = data_array.year
	y_column = data_array.annual_mean
	sigma_column = np.ones(len(x_column))/8
	if add_gaussian_error:
		for i in range(len(sigma_column)):
			sigma_column[i] = np.random.normal(loc=0, scale=100.5)

	return {'x': x_column, 'y': y_column, 'sigma': sigma_column}
	

def make_plot(x_array, y_array, sigma_array, y_fit_array, bootstrap_param_results,
	actual_params):

	font = {
		'weight' : 'bold',
		'size'   : 14}

	plt.rc('font', **font)

	fig = plt.figure(figsize=(14,8), )

	ax1 = fig.add_subplot(121)
	ax1.grid()
	ax1.set_xlabel('Year')
	ax1.set_ylabel('temperature difference (C)')
	ax1.errorbar(x_array, y_array, sigma_array, label='')
	ax1.plot(x_array, y_fit_array, label='fit')
	ax1.plot(x_array, y_array, 'ko', label='data')
	plt.legend()
	plt.title('Fitting the original data')

	ax2 = fig.add_subplot(122)
	ax2.grid()
	ax2.set_xlabel('$a_0$ (y-intercept)')
	ax2.set_ylabel('$a_1$ (slope)')
	ax2.scatter(bootstrap_param_results.T[0], bootstrap_param_results.T[1],
		 label='bootstrap results')
	ax2.scatter([actual_params[0]], [actual_params[1]], color='r', label='Actual best fit')

	plt.title('Scatterplot of best fit from bootstrap')
	plt.legend()
	plt.tight_layout()
	plt.suptitle('Change in global temperature')

	plt.show()

# Bootstrapping method: return an equal-length array of randomly-picked items from original.
def gen_set_of_randomly_selected_data(input_x, input_y):
	output_xy = np.zeros([len(input_y),2])
	
	for i in range(len(input_y)):
		index = np.random.randint(len(input_y))
		output_xy[i] = [input_x[index], input_y[index]]
		
	return output_xy


# Manaully 'measure' sigma for a list of numbers.
def find_dist_from_middle_to_68th_percentile(a_list_array):
	sorted_array = np.sort(a_list_array)
	median_element = np.median(a_list_array)
	index_at_68th = np.round(0.68 * len(a_list_array))
	element_at_68th = sorted_array[index_at_68th]

	return element_at_68th - median_element

# Make linear fit of the data and plot the fit over the data.
def do_part_3():

	temperature_data = np.recfromcsv('temperature_data.csv', delimiter=',')
	x_y_sigma = take_data_make_x_y_sigma(temperature_data)
	fit_params = calculate_fit_params(x_y_sigma['x'], x_y_sigma['y'],
		x_y_sigma['sigma'])
	print 'polynomial chi_squared reduced: {}'.format(fit_params['chi_squared'])
	print 'best fit polynomial params: {}'.format(fit_params['poly_params'])
	y_fit_array = make_poly(fit_params['poly_params'], x_y_sigma['x'])
	bootstrap_param_results = []
	# Compute best fit parameters for N sets of bootstrapped data.
	for i in range(1000):
		bootstrap_xy = gen_set_of_randomly_selected_data(x_y_sigma['x'], x_y_sigma['y'])
		bootstrap_fit_params = calculate_fit_params(bootstrap_xy.T[0], bootstrap_xy.T[1],
			x_y_sigma['sigma'])
		bootstrap_param_results.append(bootstrap_fit_params['poly_params'])

	# convert to an array. Yes, sloppy programming. 
	bootstrap_param_results = np.array(bootstrap_param_results)
	bootstrapped_a0_array = bootstrap_param_results.T[0]
	bootstrapped_a1_array = bootstrap_param_results.T[1]
	print "'Sigma' for a0: {}".format(find_dist_from_middle_to_68th_percentile(
		bootstrapped_a0_array))
	print "'Sigma' for a1: {}".format(find_dist_from_middle_to_68th_percentile(
		bootstrapped_a1_array))


	make_plot(x_y_sigma['x'], x_y_sigma['y'], x_y_sigma['sigma'], y_fit_array,
		bootstrap_param_results, fit_params['poly_params'])


# Do the same as above but now with added Gaussian errors.
def do_part_4():

	temperature_data = np.recfromcsv('temperature_data.csv', delimiter=',')
	x_y_sigma = take_data_make_x_y_sigma(temperature_data, add_gaussian_error=True)
	fit_params = calculate_fit_params(x_y_sigma['x'], x_y_sigma['y'],
		x_y_sigma['sigma'])
	print 'polynomial chi_squared reduced: {}'.format(fit_params['chi_squared'])
	print 'best fit polynomial params: {}'.format(fit_params['poly_params'])
	y_fit_array = make_poly(fit_params['poly_params'], x_y_sigma['x'])
	bootstrap_param_results = []
	# Compute best fit parameters for N sets of bootstrapped data.
	for i in range(1000):
		bootstrap_xy = gen_set_of_randomly_selected_data(x_y_sigma['x'], x_y_sigma['y'])
		bootstrap_fit_params = calculate_fit_params(bootstrap_xy.T[0], bootstrap_xy.T[1],
			x_y_sigma['sigma'])
		bootstrap_param_results.append(bootstrap_fit_params['poly_params'])

	# convert to an array. Yes, sloppy programming. 
	bootstrap_param_results = np.array(bootstrap_param_results)
	bootstrapped_a0_array = bootstrap_param_results.T[0]
	bootstrapped_a1_array = bootstrap_param_results.T[1]
	print "'Sigma' for a0: {}".format(find_dist_from_middle_to_68th_percentile(
		bootstrapped_a0_array))
	print "'Sigma' for a1: {}".format(find_dist_from_middle_to_68th_percentile(
		bootstrapped_a1_array))


	make_plot(x_y_sigma['x'], x_y_sigma['y'], x_y_sigma['sigma'], y_fit_array,
		bootstrap_param_results, fit_params['poly_params'])




def main():
	do_part_3()
	do_part_4()

main()
