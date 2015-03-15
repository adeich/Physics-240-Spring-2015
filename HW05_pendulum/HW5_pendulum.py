# Compute the trajectory of a pendulum using different ODE algorithms:
# Euler, Euler-Cromer, and Midpoint.

import numpy as np
import matplotlib.pyplot as plt

# Returns time and position arrays.
def compute_trajectory(dt, final_t, theta0=np.pi/6, algorithm=None):

	# Compute acceleration as function of position.
	def compute_a(currTheta, g=9.8, L=1.0):
		return -g/L * currTheta

	# Create the t, theta, and omega arrays.
	t_array = np.arange(0, final_t, dt)
	theta_array = np.empty(len(t_array))
	omega_array = np.empty(len(t_array))

	# initialize first elements of theta and omega.
	theta_array[0] = theta0
	omega_array[0] = 0

	if algorithm == 'Euler-Cromer':
		# Euler-Cromer method right here. Notice that omega_i computed first.
		for i in range(1, len(t_array)):
			omega_array[i] = omega_array[i-1] + dt * compute_a(theta_array[i-1])
			theta_array[i] = theta_array[i-1] + dt * omega_array[i] 
	elif algorithm == 'Midpoint':
		# Midpoint method.
		for i in range(1, len(t_array)):
			omega_array[i] = omega_array[i-1] + dt * compute_a(theta_array[i-1])
			theta_array[i] = theta_array[i-1] + dt * (0.5) * (omega_array[i] + omega_array[i-1]) 
	elif algorithm == 'Euler':
		for i in range(1, len(t_array)):
			theta_array[i] = theta_array[i-1] + dt * omega_array[i-1]
			omega_array[i] = omega_array[i-1] + dt * compute_a(theta_array[i-1])
	else:
		raise BaseException('Invalid algorithm! "{}"'.format(algorithm))
	
	# return t_array and theta_array.
	return {'t': t_array, 'theta': theta_array}	


def plot_all(dt, final_t):

	euler_traj = compute_trajectory(dt, final_t, theta0=np.pi/6, algorithm='Euler')
	euler_cromer_traj = compute_trajectory(dt, final_t, 
		theta0=np.pi/6, algorithm='Euler-Cromer')
	midpoint_traj = compute_trajectory(dt, final_t, theta0=np.pi/6, algorithm='Midpoint')
 
	# Create the plot object.
	fig = plt.figure(facecolor='w')
	ax = fig.add_subplot(111)
	ax.set_title('Pendulum Motion, dt = {}s\nAaron Deich'.format(dt))
	ax.set_xlabel('time (seconds)'); ax.set_ylabel('theta (radians)')

	# Add the three different data sets to the plot.
	ax.plot(euler_traj['t'], euler_traj['theta'], 'c', label='Euler')
	ax.plot(euler_cromer_traj['t'], euler_cromer_traj['theta'], label='Euler-Cromer')
	ax.plot(midpoint_traj['t'], midpoint_traj['theta'], '--', label='Midpoint')

	ax.legend(loc='top right')
	plt.show()

# Calculate the period of the pendulum swing using Euler-Cromer. This function
# actually runs the pendulum for N cycles, then returns the avg of those periods.
def calc_avg_T(dt, theta0, N_cycles, algorithm):
	currTheta = None
	currOmega = None
	prevTheta = theta0
	prevOmega = 0
	half_periods = []
	prev_time = [0]
	total_time = 0

	# Compute acceleration as function of position.
	def compute_a(Theta, g=9.8, L=1.0):
		return -g/L * Theta

	# go until N cycles have passed.
	while len(half_periods)< 2*N_cycles:
		if algorithm == 'Euler-Cromer':
			currOmega = prevOmega + dt * compute_a(prevTheta)
			currTheta = prevTheta + dt * currOmega 
		elif algorithm == 'Midpoint':
			currOmega = prevOmega + dt * compute_a(prevTheta)
			currTheta = prevTheta + 0.5 * dt * (prevOmega + currOmega) 
		elif algorithm == 'Euler':
			currOmega = prevOmega + dt * compute_a(prevTheta)
			currTheta = prevTheta + dt * prevOmega 

		# If theta has just passed 0 in either direction.
		if currTheta * prevTheta < 0:
			half_periods.append(total_time - prev_time[-1])
			prev_time.append(total_time)
		prevTheta = currTheta; prevOmega = currOmega
		total_time += dt

	full_periods = []
	if not len(half_periods) % 2 == 0:
		raise BaseException("Not multiple of 2!")
	for i in range(0, len(half_periods), 2):
		full_periods.append(half_periods[i] + half_periods[i+1]) 
	
	return {'avgT': np.average(full_periods),
			'uncertainty': np.std(full_periods) / np.sqrt(N_cycles)}

# Find starting angle where small-angle approximations break down.
def find_breakdown_theta():

	# Small-angle period of pendulum
	def first_order_approx_T(theta_max, L=1.0, g=9.8):
		return 2 * np.pi * np.sqrt(L/g)
	# Same as above but also with quadratic term.
	def second_order_approx_T(theta_max, L=1.0, g=9.8):
		return 2 * np.pi * np.sqrt(L/g) * (1 + theta_max**2/16.)
	
	def calc_error(true_number, test_number):
		return (test_number - true_number) / (true_number)

	theta0_array = np.linspace(np.pi/8, np.pi/2, 10)
	#first_approx_error_array = np.array([calc_error(first_order_approx_T(theta0), 
#		calc_avg_T(0.01, theta0, 10., 'Euler-Cromer')['avgT']) for theta0 in theta0_array])

	for theta0 in theta0_array:
		print calc_error(first_order_approx_T(theta0), calc_avg_T(0.01, theta0, 10., 'Euler-Cromer')['avgT']) 


	for theta0 in theta0_array:
		print theta0, calc_avg_T(0.01, theta0, 10., 'Euler-Cromer')['avgT'], second_order_approx_T(theta0)


# Compute pendulum motion via three different methods over a given time
# interval and plot all three curves to the same graph.
# plot_all(0.1, 10.0)

print calc_avg_T(0.001, np.pi/4, 10., 'Euler-Cromer')
find_breakdown_theta()
