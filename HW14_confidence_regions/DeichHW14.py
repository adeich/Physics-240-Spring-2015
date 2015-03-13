import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

def make_poly(a,x) : # polynomial model of vector x with parameters a
	#print 'a:{}'.format(a)
	M = len(a)-1
	y_mod = np.zeros(len(x)) 
	for j in range(M+1) :
		y_mod += a[j] * x**j 
	return y_mod
 	

def calc_poly_least_squares_params(x_array, y_array, uncertainty_array, degree_polynomial):

	# pollsf() written by A. Romanowsky.
	# polynomial least squares fit function, M is polynomial order
	def pollsf(x,y,dy,M):
		b, A = np.zeros(len(x)), np.zeros((len(x),M+1))
		# design matrix :
		for i in range(len(x)) :
			b[i] = y[i]/dy[i]
			for j in range(M+1) :
				A[i,j] = x[i]**j/dy[i]
		C = np.linalg.inv(np.dot(A.T,A))  # correlation matrix
		J_arr = np.dot(C,A.T)
		a = np.dot(J_arr,b)
		da = np.sqrt(np.diag(C))
		y_mod = make_poly(a,x)
		chi2_red = sum((y_mod-y)**2/dy**2) / (len(x)-(M+1))
		return a, da, chi2_red

	a, da, chi2_red = pollsf(x=x_array,
		y=y_array,
		dy=uncertainty_array,
		M=degree_polynomial)

	return {'poly_params': a, 'chi_squared': chi2_red}


# Basically the same as Romanowsky's linreg()?
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
		'chi_squared': chi_squared}


# custom function for preparing temperature data from file. Adds faked uncertainty.
def take_data_make_x_y_sigma(data_array):
	x_column = data_array.year
	y_column = data_array.annual_mean
	sigma_column = np.ones(len(x_column))/8

	return {'x': x_column, 'y': y_column, 'sigma': sigma_column}
	

def make_plot(x_array, y_array, sigma_array, y_fit_array):


	fig = plt.figure(figsize=(14,8))

	ax1 = fig.add_subplot(111)
	ax1.grid()
	ax1.set_xlabel('Year')
	ax1.set_ylabel('temperature difference (C)')
#	ax1.set_ylim([-5e-8,5e-8])
#	ax1.set_xlim([pc['V'],0])
	ax1.errorbar(x_array, y_array, sigma_array, label='')
	ax1.plot(x_array, y_fit_array, label='fit')
	ax1.plot(x_array, y_array, 'ko', label='data')
	plt.legend()

#	x_curve_array = np.linspace(x_array[0], x_array[-1], 100)
#	y_curve_array = x_curve_array**fit_data['slope'] * np.exp(fit_data['intercept'])

#	ax2 = fig.add_subplot(122)
#	ax2.grid()
#	ax2.set_xlabel('Year')
#	ax2.set_ylabel('temperature (C)')
#	ax2.set_ylim([-5e-8,5e-8])
#	ax2.set_xlim([pc['V'],0])
#	ax2.plot(x_array, y_array, 'ko', label='temperature (C)')
#	ax2.plot(x_array, y_array,  label='temperature (C)')
#	ax2.plot(x_curve_array, y_curve_array,  label='temperature (C)')
	plt.title('Change of Mean Global Temperature (C)')
	plt.legend()

	plt.show()


# Make and polynomial fit of the data and plot the fit over the data.
def do_part_1():

	temperature_data = np.recfromcsv('CommaData.csv', delimiter=',')
	x_y_sigma = take_data_make_x_y_sigma(temperature_data)
	fit_params = calc_poly_least_squares_params(x_y_sigma['x'], x_y_sigma['y'],
		x_y_sigma['sigma'], degree_polynomial=3)
	print 'polynomial chi_squared reduced: {}'.format(fit_params['chi_squared'])
	print 'best fit polynomial params: {}'.format(fit_params['poly_params'])
	y_fit_array = make_poly(fit_params['poly_params'], x_y_sigma['x'])

	make_plot(x_y_sigma['x'], x_y_sigma['y'], x_y_sigma['sigma'], y_fit_array)


# Design a non-linear model and fit to the data.
def do_part_2():

	# model function of temperature data. Intended for scipy.optimize.curve_fit().
	def model_function(x, a_slope, a_y_int, a_freq, a_amplitude):
		return a_slope * x + a_amplitude * np.sin(a_freq * x) + a_y_int
	
	temperature_data = np.recfromcsv('CommaData.csv', delimiter=',')
	x_y_sigma = take_data_make_x_y_sigma(temperature_data)
	optimal_params_list, covariant_params = scipy.optimize.curve_fit(model_function,
		x_y_sigma['x'], x_y_sigma['y'], p0=[9.03838147e-03, 1, 1e-1, 1]) # slope, y-int, freq, ampl
	print 'params list: {}.'.format(optimal_params_list)
	y_fit_array = model_function(x_y_sigma['x'], *optimal_params_list)

	make_plot(x_y_sigma['x'], x_y_sigma['y'], x_y_sigma['sigma'], y_fit_array)

def main():
	do_part_1()
	do_part_2()	

main()
