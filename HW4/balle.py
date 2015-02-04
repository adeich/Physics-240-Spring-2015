import numpy as np
import matplotlib.pyplot as plt

def compute_exact_trajectory_arrays(physical_param_dict, initial_conditions_dict):
	theta0 = initial_conditions_dict['theta0']; v0 = initial_conditions_dict['v0']
	g = physical_param_dict['g']; dt = initial_conditions_dict['dt']
	v0x = np.cos(theta0) * v0; v0y = np.sin(theta0) * v0
	flight_time = 2 * v0y / g
	time_array = np.arange(0, flight_time, dt)
	x_array = v0x * time_array 
	y_array = -0.5 * g * time_array**2 + v0y * time_array  

	return np.array([x_array, y_array]).T


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
			return np.array([0, -physical_param_dict['g']]) + (1/m) * compute_force_air_resistance(currV, physical_param_dict)
			

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
		currA = compute_acceleration_vector(currV, physical_param_dict, air_resistance=False)
		currR, currV = compute_next_R_and_V(currR, currV, currA, dt)
		list_of_R.append(currR); list_of_V.append(currV)
	

	position_array = np.array(list_of_R); velocity_array = np.array(list_of_V)

	return position_array 

def plot_trajectories(approx_xy=None, exact_xy=None, air_resistance=False):
	
	plt.figure(facecolor='w')

	if approx_xy is not None:
		plt.plot(approx_xy.T[0], approx_xy.T[1], 'b-', label="Euler's method")
	if exact_xy is not None:
		plt.plot(exact_xy.T[0], exact_xy.T[1], 'g--', label='Exact solution')
	if not air_resistance:
		plt.title('Football Trajectory, No Air Resistance (equal axes scale)')
	else:
		plt.title('Football Trajectory, with Air Resistance (equal axes scale)')
	plt.xlabel('x (meters)'); plt.ylabel('y (meters)')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.legend(loc='top right')

	#plt.savefig('HW4.png')
	plt.show()


def main(v0, theta0, dt, air_resistance=False):
	physical_param_dict = {
	'm': 0.43, # kg
	'Cd': 100.1,  # 0.05, # unitless
	'ro': 1.2, # kg m^-3
	'A': 0.013, # m^2
	'g': 9.8 # m s^2
	}
	initial_conditions_dict = {
	'v0': v0,
	'theta0': theta0,
	'dt': dt
	}

	approx_pos_array = compute_approx_trajectory_arrays(physical_param_dict, 
		initial_conditions_dict, air_resistance)
	exact_pos_array = compute_exact_trajectory_arrays(physical_param_dict,
		initial_conditions_dict)

	plot_trajectories(approx_xy=approx_pos_array, exact_xy=exact_pos_array, air_resistance=air_resistance)


main(10, np.pi/4, 0.01, air_resistance=False)
main(10, np.pi/4, 0.01, air_resistance=True)
