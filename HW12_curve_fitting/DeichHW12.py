import numpy as np
import matplotlib.pyplot as plt


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


# custom function for processing temperature data.
def take_data_make_x_y_sigma(data_array):
	x_column = data_array.year
	y_column = data_array.annual_mean
	sigma_column = np.ones(len(x_column))/5

	return {'x': x_column, 'y': y_column, 'sigma': sigma_column}
	

def make_plot(x_array, y_array, sigma_array, fit_data, log_fit_data):

	# in log plot.
	x_fit_array = np.linspace(np.min(np.log(x_array)), np.max(np.log(x_array)), 100)
	y_fit_array = log_fit_data['slope'] * x_fit_array + log_fit_data['intercept']

	fig = plt.figure(figsize=(14,8))

	ax1 = fig.add_subplot(121)
	ax1.grid()
	ax1.set_xlabel('Year')
	ax1.set_ylabel('log(temp)')
#	ax1.set_ylim([-5e-8,5e-8])
#	ax1.set_xlim([pc['V'],0])
	ax1.errorbar(x_array, np.log(y_array), np.log(sigma_array), label='')
	ax1.plot(x_fit_array, y_fit_array, label='linear fit')
	plt.title('Log of Mean Global Temperature (C)')
	plt.legend()

	x_curve_array = np.linspace(x_array[0], x_array[-1], 100)
	y_curve_array = x_curve_array**fit_data['slope'] * np.exp(fit_data['intercept'])

	ax2 = fig.add_subplot(122)
	ax2.grid()
	ax2.set_xlabel('Year')
	ax2.set_ylabel('temperature (C)')
#	ax2.set_ylim([-5e-8,5e-8])
#	ax2.set_xlim([pc['V'],0])
	ax2.plot(x_array, y_array, 'ko', label='temperature (C)')
	ax2.plot(x_array, y_array,  label='temperature (C)')
	ax2.plot(x_curve_array, y_curve_array,  label='temperature (C)')
	plt.title('Mean Global Temperature (C)')
	plt.legend()



	plt.show()



def main():

	temperature_data = np.recfromcsv('CommaData.csv', delimiter=',')
	x_y_sigma = take_data_make_x_y_sigma(temperature_data)
	fit_data = calculate_fit_params(x_y_sigma['x'], x_y_sigma['y'], x_y_sigma['sigma'])
	log_fit_data = calculate_fit_params(x_y_sigma['x'], 
		np.log(x_y_sigma['y']), np.log(x_y_sigma['sigma']))

	print fit_data

	make_plot(x_y_sigma['x'], x_y_sigma['y'], x_y_sigma['sigma'], fit_data, log_fit_data)



main()
