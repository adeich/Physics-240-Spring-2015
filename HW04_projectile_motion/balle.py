import numpy as np
import matplotlib.pyplot as plt


# Computes the x and y arrays for the exact vacuum trajectory solution.
def compute_exact_trajectory_arrays(physical_param_dict, initial_conditions_dict):
	theta0 = initial_conditions_dict['theta0']; v0 = initial_conditions_dict['v0']
	g = physical_param_dict['g']; dt = initial_conditions_dict['dt']
	v0x = np.cos(theta0) * v0; v0y = np.sin(theta0) * v0
	flight_time = 2 * v0y / g
	time_array = np.arange(0, flight_time, dt)
	x_array = v0x * time_array 
	y_array = -0.5 * g * time_array**2 + v0y * time_array  

	return {'flight_time': flight_time,
		'theta': initial_conditions_dict['theta0'],
		'dt': initial_conditions_dict['dt'],
		'x_distance': x_array[len(x_array) - 1],  
		'xy_array': np.array([x_array, y_array]).T
		}


# Numerical trajectory for object, from time it leaves ground to time it hits ground.
def compute_approx_trajectory_arrays(physical_param_dict, 
	initial_conditions_dict, air_resistance=False):

	### AUXILIARY FUNCTIONS ###

	# Calculates the acceleration vector for a given position and velocity.
	def compute_acceleration_vector(currV, physical_param_dict, air_resistance=False):

		# returns a force vector. 'pd' short for physical_param_dict.
		def compute_force_air_resistance(currV, pd):
			force_mag = - 0.5 * pd['Cd'] * pd['A'] * pd['ro'] * np.linalg.norm(currV)
			force_vec = force_mag * currV
			return force_vec # kg m s^2
	
		if not air_resistance:
			return np.array([0, -physical_param_dict['g']])
		else:
			mass = physical_param_dict['m']
			return np.array([0, -physical_param_dict['g']]) + (1/mass) * compute_force_air_resistance(currV, physical_param_dict)
			

	# Euler's method. R, V, and A are each 3-vectors for pos, vel, and acc. 
	def compute_next_R_and_V(currR, currV, currA, dt):
		nextR = currR + currV * dt
		nextV = currV + currA * dt
		return nextR, nextV

	### MAIN PROCESS ###
	
	# from initial conditions, determine first R and V vectors.
	theta0 = initial_conditions_dict['theta0']; v0 = initial_conditions_dict['v0']
	currR = np.array([0,0])
 	currV = np.array([np.cos(theta0) * v0, np.sin(theta0) * v0])
	dt = initial_conditions_dict['dt']

	list_of_R = [currR]
	list_of_V = [currV]

	# while height is greater than 0, find next R and V and add them to list.
	while currR[1] >= 0:
		currA = compute_acceleration_vector(currV, physical_param_dict, air_resistance)
		currR, currV = compute_next_R_and_V(currR, currV, currA, dt)
		list_of_R.append(currR); list_of_V.append(currV)
	

	position_array = np.array(list_of_R); velocity_array = np.array(list_of_V)

	return {'flight_time': dt * len(position_array),
		'theta': initial_conditions_dict['theta0'],
		'dt': initial_conditions_dict['dt'],
		'x_distance': position_array[len(position_array)-1][0],
		'xy_array': position_array 
		}

def plot_trajectories(approx_xy=None, exact_xy=None, air_resistance=False):

	exact_xy_array = exact_xy['xy_array']
	approx_xy_array = approx_xy['xy_array']
	
	fig = plt.figure(facecolor='w')

	# add time and distance to landing spot.
	approx_timestring = 'Euler\'s Method:\n range: {}m \n t: {}s'.format(approx_xy['x_distance'], approx_xy['flight_time'])
	exact_timestring =  'Exact solution:\n range: {}m \n t: {}s'.format(exact_xy['x_distance'], exact_xy['flight_time'])
	ax = fig.add_subplot(111)
	sTimeStep = 'time step: {} s'.format(exact_xy['dt'])
	sTheta = 'Initial angle: {0:.2f} deg'.format(exact_xy['theta'] * (180./np.pi))


	sWholeString = '{}\n{}\n{}\n{}'.format(sTimeStep, sTheta,
		exact_timestring, approx_timestring)
	ax.annotate(sWholeString, xy=(approx_xy['x_distance'], 0), xytext=(1, 5))

	# horizontal line representing the ground.
	ax.axhline(y=0, xmin=0, xmax=1, color='black')
	if approx_xy is not None:
		ax.plot(approx_xy_array.T[0], approx_xy_array.T[1], 'b-', label="Euler's method")
	if exact_xy is not None:
		ax.plot(exact_xy_array.T[0], exact_xy_array.T[1], 'g--', label='Exact solution\n(no air)')
	if not air_resistance:
		ax.set_title('Football Trajectory, No Air Resistance')
	else:
		ax.set_title('Football Trajectory, with Air Resistance')
	ax.set_xlabel('x (meters)'); ax.set_ylabel('y (meters)')
	ax.set_aspect('equal', adjustable='box')
	ax.legend(loc='top left')

	#plt.savefig('HW4.png')
	plt.show()


def main(v0, theta0, dt, air_resistance=False):
	physical_param_dict = {
	'm': 0.43, # kg
	'Cd': 0.05,  # 0.05, # unitless
	'ro': 1.2, # kg m^-3
	'A': 0.013, # m^2
	'g': 9.8 # m s^2
	}
	initial_conditions_dict = {
	'v0': v0,
	'theta0': theta0,
	'dt': dt
	}

	# Calculate the trajectory with Euler's method.
	approximation_data = compute_approx_trajectory_arrays(physical_param_dict, 
		initial_conditions_dict, air_resistance)
	# Calculate with the analytical method (no air resistance).
	exact_data = compute_exact_trajectory_arrays(physical_param_dict,
		initial_conditions_dict)
	
	# Pass the respective trajectory arrays to the plotting function.
	plot_trajectories(approx_xy=approximation_data, exact_xy=exact_data, air_resistance=air_resistance)


main(30., np.pi/6, 0.01, air_resistance=False)
# main(30., np.pi/6, 0.01, air_resistance=True)
