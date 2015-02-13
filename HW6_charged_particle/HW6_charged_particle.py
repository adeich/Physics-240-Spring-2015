import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# returns vector. All input-vectors must be of same dimension.
def calc_lorentz(q, v_vec, E_vec, B_vec):
	return q * (E_vec + np.cross(v_vec, B_vec))

# Calculate particle trajectory. Assumes constant E and B fields.
def calc_particle_traj(dt, final_t, q, mass, x0_vec, v0_vec, E_vec, B_vec):

	# Create array of all timesteps. 
	t_array = np.arange(0, final_t, dt)

	# Create empty position array.
	position_array = np.empty([len(t_array), 3])

	# initialize 
	current_v_vec = v0_vec
	position_array[0] = x0_vec

	# Euler-Cromer method, chosen because in this problem we need to know
	# in order: nth velocity -> nth acceleration -> nth position. Thus the
	# Verlet and leap frog methods won't work very easily here.
	for i in range(1, len(t_array)):

		# acceleration = force / mass
		current_a_vec = calc_lorentz(q, current_v_vec, E_vec, B_vec) / mass
		current_v_vec = current_v_vec + dt * current_a_vec
		position_array[i] = position_array[i-1] + dt * current_v_vec

	return {'t_array': t_array, 'position': position_array}


def produce_plot_title(v0_vec, E_vec, B_vec):

	# make nice scientific notation strings.
	def convert_arrays_to_sci(input_array):
	
		def choose_format(number):
			if number == 0:
				return '0'
			else:
				return '{:.1e}'.format(number)
	
		stringified_array = [choose_format(i) for i in input_array]
		return '[{}]'.format(', '.join(stringified_array))

	v0_str = convert_arrays_to_sci(v0_vec)
	E_str = convert_arrays_to_sci(E_vec)
	B_str = convert_arrays_to_sci(B_vec)
	
	return 'Lorentz Force\nv0:{} m/s\nE:{} N/C\nB:{} T'.format(v0_str, E_str, B_str)


def measure_drift_velocity(trajectory_dict):
	position = trajectory_dict['position']
	t_array = trajectory_dict['t_array']


	# first, record where the maxima occur in X and in t (assumes helix is in x-y plane).
	maxima_locations = []
	maxima_times = []
	for i in range(1, len(position)-1):
		# if we've just passed X=0 in either direction.
		if (position[i][0] - position[i-1][0] < 0) and (position[i+1][0] - position[i][0] > 0):
			maxima_locations.append(position[i])
			maxima_times.append(t_array[i])
	maxima_locations = np.array(maxima_locations)
	maxima_times = np.array(maxima_times)

	# second, create a list of velocities between all maxima.
	drift_velocities = np.empty([len(maxima_locations)-1, 3])
	for i in range(0, len(drift_velocities)):
		dt = maxima_times[i+1] - maxima_times[i]
		drift_velocities[i] = (maxima_locations[i+1] - maxima_locations[i]) / dt

	# third, find the average and standard deviation of these velocities.
	average_v = np.mean(drift_velocities, axis=0)
	speeds = np.array([np.linalg.norm(v) for v in drift_velocities])
	std_speeds = np.std(speeds)
	mean_speeds = np.mean(speeds)


	return {'n_maxima': len(maxima_locations), 'average_v_vec': average_v,
		 'std_speeds': std_speeds, 'mean_speed': mean_speeds, 
			'list_of_v': drift_velocities, 'maxima': maxima_locations,
			'maxima_times': maxima_times}

# Run trajectory function and make 3D plot (x, y, z).
def make_3d_plot(dt, final_t, q, mass, v0_vec, x0_vec, E_vec, B_vec):
	trajectory_dict = calc_particle_traj(dt, final_t, q, mass, x0_vec, v0_vec, E_vec, B_vec)
	t_array = trajectory_dict['t_array']; position_array = trajectory_dict['position']
	
	drift_dict = measure_drift_velocity(trajectory_dict)
	drift_loc = drift_dict['maxima']
	drift_times = drift_dict['maxima_times']

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	fig = plt.figure(figsize=(8,8), dpi=100, facecolor='w')
	ax = fig.gca(projection='3d')
	ax.plot(position_array.T[0], position_array.T[1], t_array)
	ax.plot(drift_loc.T[0], drift_loc.T[1], drift_times)
	ax.view_init(21,-60)
	plt.xlabel('$x$ meters', size=14);	plt.ylabel('$y$ meters', size=14);
	ax.set_zlabel('$t$ seconds', size=14)
	plt.ticklabel_format(style='sci', axis='both', scilimits=(1,3))
	plt.suptitle(produce_plot_title(v0_vec, E_vec, B_vec), fontsize=14)
	plt.show()

def make_2d_plot(dt, final_t, q, mass, v0_vec, E_vec, B_vec):
	trajectory_dict = calc_particle_traj(dt, final_t, q, mass, v0_vec, E_vec, B_vec)
	t_array = trajectory_dict['t_array']; position_array = trajectory_dict['position']
	
	#ax = fig.gca(projection='3d')

	plt.plot(position_array.T[0], position_array.T[1])
#	ax.view_init(21,-60)
	plt.grid(True)
	plt.xlabel('$x$', size=18);	plt.ylabel('$y$')	
	plt.ticklabel_format(style='sci', axis='both', scilimits=(1,3))
	plt.show()


# global input constants.
pc = {'mass_proton': 1.67e-19, # kg
	'q_proton': 1.60e-19, # C
	'x0': np.array([-3e3, 0, 0]), # m
	'v0': np.array([0, 1.5e3, 0]), # m/s
	'E_vec': np.array([1e2, 0, 0]), # ?
	'B_vec': np.array([0, 0, 0.5]) # T
	}

def main():

	trajectory_dict = calc_particle_traj(dt=0.01, final_t=100., 
		q=pc['q_proton'], mass=pc['mass_proton'], v0_vec=pc['v0'], 
		x0_vec=pc['x0'], E_vec=pc['E_vec'], B_vec=pc['B_vec'])
	print measure_drift_velocity(trajectory_dict) 

	make_3d_plot(dt=0.01, final_t=100., 
		q=pc['q_proton'], mass=pc['mass_proton'], v0_vec=pc['v0'], 
		x0_vec=pc['x0'], E_vec=pc['E_vec'], B_vec=pc['B_vec'])




main()
