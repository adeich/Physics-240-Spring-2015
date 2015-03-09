import numpy as np
import matplotlib.pyplot as plt

# A template for writing to and reading from csv file.
global_column_ordering = ['theta0', 'average_period', 'uncertainty'] 

# physical constants. l is length of pendulum.
pc = {'g': 9.8, # m/s^2
	'l': 1.0 # m
	}

# Based off a function I wrote for HW #5: pendulum trajectory with Euler Cromer.
# Calculate the period of the pendulum swing using Euler-Cromer. This function
# actually runs the pendulum for N cycles, then returns the avg of those periods.
def calc_period_stats(dt, theta0, N_cycles):
	theta0 = np.abs(theta0) # make sure theta0 is positive
	currTheta = None
	currOmega = None
	prevTheta = theta0
	prevOmega = 0
	T_completion_times = []
	total_time = 0

	# Compute acceleration as function of position.
	def compute_a(Theta):
		return -pc['g']/pc['l'] * np.sin(Theta)

	# Euler-Cromer method. Go until N cycles have passed.
	while len(T_completion_times) < N_cycles:
		currOmega = prevOmega + dt * compute_a(prevTheta)
		currTheta = prevTheta + dt * currOmega 

		# Check to see if we're entering the next period, i.e. ask if 
		# omega has just passed 0 and is heading into the positive direction. 
		if ((currOmega * prevOmega < 0) and (currOmega < 0)):
			T_completion_times.append(total_time)

		prevTheta = currTheta; prevOmega = currOmega
		total_time += dt

	# Compute period durations.
	period_duration_measurements = [T_completion_times[i+1] - T_completion_times[i] for 
		i in range(len(T_completion_times)-1)]

	return {'average_period': np.average(period_duration_measurements),
			'uncertainty': np.std(period_duration_measurements) / np.sqrt(N_cycles),
			'small-angle-T': 2 * np.pi * np.sqrt(pc['l']/pc['g']),
			'period_duration_measurements': period_duration_measurements,
			'theta0': theta0}


# Runs calc_period_stats() over range of theta0. Takes several seconds.
def compute_periods_for_range_of_theta0(theta0_min, theta0_max, theta_number):
	theta0_array = np.linspace(theta0_min, theta0_max, theta_number)
	period_statistics = []
	for theta0 in theta0_array:
		period_statistics.append(calc_period_stats(dt=0.01, theta0=theta0, N_cycles=10))
		
	return period_statistics

def save_periods_data_to_file(period_dict_list, sFilename):
	with open(sFilename, 'w') as f:
		f.write(','.join(global_column_ordering) + '\n') # write the header line
		for period_dict in period_dict_list:
			printingData= []
			for columnName in global_column_ordering:
				printingData.append(str(period_dict[columnName]))
			f.write(','.join(printingData)); f.write('\n')

def read_period_data_file(sFilename):
	with open(sFilename, 'r') as f:
		period_rec_array = np.recfromcsv(f)
	
	return period_rec_array


def make_poly(a,x) : # polynomial model of vector x with parameters a
	print 'a:{}'.format(a)
	M = len(a)-1
	y_mod = np.zeros(len(x)) 
	for j in range(M+1) :
		y_mod += a[j] * x**j 
	return y_mod
 	

def calc_best_sqr_fit_params(period_data_array):

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

	a, da, chi2_red = pollsf(x=period_data_array.theta0,
		y=period_data_array.average_period,
		dy=period_data_array.uncertainty,
		M=2)

	return {'poly_params': a, 'chi_squared': chi2_red}

def generate_fit_array(polynomial_params, x_data_array):
	x_fit_array = np.linspace(np.min(x_data_array), np.max(x_data_array), 100)
	y_fit_array = make_poly(polynomial_params, x_fit_array)

	return {'x_array': x_fit_array, 'y_array': y_fit_array}

def pendulum_second_order_expansion(theta0):
	return 2 * np.pi * np.sqrt(pc['l']/pc['g']) * (1. + theta0**2 /16.)	

def make_plot(period_data_array, fit_data_dict, best_fit_params_list):
	polynomial_formula = 'Best fit: $T(a_0)={:.2f}a_0^2 + {:.2f}a_0 + {:.2f}$'.format(
		*best_fit_params_list['poly_params'])

	# Create the plot object.
	fig = plt.figure(facecolor='w', figsize=(14,8))
	ax = fig.add_subplot(111)
	ax.set_title('Period of a pendulum vs. starting angle\n\n{}'.format(polynomial_formula))
	ax.set_xlabel('Starting angle $a_0$ (radians)'); ax.set_ylabel('period (seconds)')

	# Add the data.
	ax.plot(fit_data_dict['x_array'], fit_data_dict['y_array'], label='best parabolic fit')
	ax.errorbar(period_data_array.theta0, period_data_array.average_period,
		 period_data_array.uncertainty, fmt='ok', label='measurements with error')
	ax.plot(fit_data_dict['x_array'], pendulum_second_order_expansion(fit_data_dict['x_array']),
		'--', label='2nd order expansion')

	ax.legend(loc='top right')
	plt.show()




def main():
	# un-comment and run these two functions to calculate and save the period data. 
	all_periods_data = compute_periods_for_range_of_theta0(theta0_min=np.pi/1000,
		theta0_max=np.pi/2, theta_number=20)
	save_periods_data_to_file(all_periods_data, 'all_data.csv')

	# read the CSV theta0-vs-period file.
	period_data_array = read_period_data_file('all_data.csv')
	best_fit_params_list = calc_best_sqr_fit_params(period_data_array)
	fit_arrays_dict = generate_fit_array(best_fit_params_list['poly_params'],
		 period_data_array.theta0)
	make_plot(period_data_array, fit_arrays_dict, best_fit_params_list)

main()
