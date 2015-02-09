# Compute the trajectory of a pendulum using different ODE algorithms:
# Euler, Euler-Cromer, and Midpoint.

import numpy as np
import matplotlib.pyplot as plt

# Returns time and position arrays.
def compute_trajectory(dt, final_t, theta0=np.pi/6, algorithm=None):

	# Compute acceleration as function of position.
	def compute_a_n(currTheta, g=9.8, L=1.0):
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
			omega_array[i] = omega_array[i-1] + dt * compute_a_n(theta_array[i-1])
			theta_array[i] = theta_array[i-1] + dt * omega_array[i] 
	elif algorithm == 'Midpoint':
		# Midpoint method.
		for i in range(1, len(t_array)):
			omega_array[i] = omega_array[i-1] + dt * compute_a_n(theta_array[i-1])
			theta_array[i] = theta_array[i-1] + dt * (0.5) * (omega_array[i] + omega_array[i-1]) 
	elif algorithm == 'Euler':
		# Midpoint method.
		for i in range(1, len(t_array)):
			theta_array[i] = theta_array[i-1] + dt * omega_array[i-1]
			omega_array[i] = omega_array[i-1] + dt * compute_a_n(theta_array[i-1])
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



plot_all(0.1, 10.0)
