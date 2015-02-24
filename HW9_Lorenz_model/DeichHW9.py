import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# physical parameters dictionary.
pc = {'sigma': 10.,
		'b': 8./3.,
		'r': 28.
	}


# Derivative function for vector X = np.array([x, y, z])
def dX_dt(pos_vec):
	# total, overboard expansion to make sure I'm not making mistakes.
	x = pos_vec[0]
	y = pos_vec[1]
	z=  pos_vec[2]
	sigma = pc['sigma']
	r = pc['r']
	b = pc['b']
	
	dx_dt = sigma * (y - x)
	dy_dt = r*x - y - x*z
	dz_dt = x*y - b*z

	return np.array([dx_dt, dy_dt, dz_dt])


# RK4 returns the next spatial position.
def calc_RK4_vec(derivative_func, pos_vec, dt):
	k1 = derivative_func(pos_vec)
	k2 = derivative_func(pos_vec + k1 * dt/2.)
	k3 = derivative_func(pos_vec + k2 * dt/2.)
	k4 = derivative_func(pos_vec + k3 * dt)
	return pos_vec + (k1 + 2. * (k2 + k3) + k4) * dt/6.

# similar form to 'rka()' from code examples.
def calc_RK4_adaptive(curr_pos_vec, initial_delta_t, max_allowable_err):
	curr_delta_t = initial_delta_t
	S1, S2 = (0.9, 4.0)
	curr_error_estimate = np.inf
	i_iteration, N_max_iterations = 0, 100
	while (curr_error_estimate > max_allowable_err) and (i_iteration < N_max_iterations):
		X_big_step = calc_RK4_vec(dX_dt, curr_pos_vec, curr_delta_t) 
		X_small_step1 = calc_RK4_vec(dX_dt, curr_pos_vec, curr_delta_t/2.) 
		X_small_step2 = calc_RK4_vec(dX_dt, X_small_step1, curr_delta_t/2.) 
		curr_error_estimate = np.max(
			np.abs(X_big_step - X_small_step2) / (np.abs(X_big_step) + np.abs(X_small_step2) + 1e-16/max_allowable_err)
			) / 2.0
		estimated_delta_t = S1 * curr_delta_t * np.abs(max_allowable_err / curr_error_estimate)**0.2
		if S1 * estimated_delta_t > S2 * curr_delta_t:
			curr_delta_t = S2 * curr_delta_t
		elif (S1 * estimated_delta_t) < (curr_delta_t / S2):
			curr_delta_t = curr_delta_t / S2
		else:
			curr_delta_t = S1 * estimated_delta_t
		
		i_iteration += 1
	
	if i_iteration == N_max_iterations:
		print "warning: adaptive RK4 did not converge"

	return X_small_step2, curr_delta_t

			

# Calculates trajectory with RK4 and non-adaptive timestep. 
def calc_trajectory_non_adaptive(dt, t0, tfinal, x0, y0, z0):
	t_array = np.arange(t0, tfinal, dt)

	# pos_array a 3xN matrix. 1st, 2nd, 3rd columns are x, y, z.
	pos_array = np.empty([len(t_array), 3])
	pos_array[0] = np.array([x0, y0, z0])

	for i in range(1, len(t_array)):
		# calculate next X vector.
		pos_array[i] = calc_RK4_vec(derivative_func=dX_dt, 
			pos_vec=pos_array[i-1], dt=dt)  
	
	print 'non-adaptive total steps: {}: tfinal: {}'.format(len(t_array), t_array[-1])
	return {'t_array': t_array, 'pos_array': pos_array}


# Calculates trajectory with RK4 and adaptive timestep. 
def calc_trajectory_adaptive(t0, tfinal, x0, y0, z0):

	# list of total-time points.
	t_list = [0]

	# pos_list is a list of 3-vectors. 1st, 2nd, 3rd elements are x, y, z.
	pos_list = [np.array([x0, y0, z0])]

	dt = 0.01 

	max_time = 5.
	start_time = time.time()

	# stop the while loop when either tfinal is reached or the program takes more
	# than max_time seconds.
	while (t_list[-1] < tfinal) and ((time.time() - start_time) < max_time):
		# calculate next X and next timestep, dt.
		next_X, dt = calc_RK4_adaptive(curr_pos_vec=pos_list[-1], initial_delta_t=dt,
			max_allowable_err=0.0001)
		pos_list.append(next_X)
		t_list.append(t_list[-1] + dt)

	print 'total steps: {}, t: {}'.format(len(pos_list), t_list[-1])
	return {'t_array': np.array(t_list), 'pos_array': np.array(pos_list)}


# Object to represent a computed trajectory.
class Trajectory:
	def __init__(self, computed_trajectory_dict):
		self.computed_trajectory_dict = computed_trajectory_dict
		self.t_array = computed_trajectory_dict['t_array']
		self.XandTarray = np.hstack([self.computed_trajectory_dict['pos_array'], 
			np.array([self.computed_trajectory_dict['t_array']]).T]) 

	def writeToFile(self, sFilename):
		np.savetxt(sFilename, self.XandTarray, delimiter=',',
			header='x, y, z, t')

	def getMinMaxTimesteps(self):
		steps_list = [self.t_array[i]-self.t_array[i-1] for i in range(1, len(self.t_array))]
		return {'min_step': np.min(steps_list), 'max_step': np.max(steps_list)}
	
	def printReport(self):
		minMaxTimeDict = self.getMinMaxTimesteps()
		print 'min timestep: {:.2e}, max timestep: {:.2e}'.format(minMaxTimeDict['min_step'], 
			minMaxTimeDict['max_step'])


# Run trajectory function and make 3D plot (x, y, z).
def make_3d_plot(trajectory_dict_non_adaptive, trajectory_dict_adaptive=None):

#	plt.rc('text', usetex=True)
#	plt.rc('font', family='serif')
	t_array = trajectory_dict_non_adaptive['t_array']
	x_array = trajectory_dict_non_adaptive['pos_array'].T[0]
	y_array = trajectory_dict_non_adaptive['pos_array'].T[1]
	z_array = trajectory_dict_non_adaptive['pos_array'].T[2]

	fig = plt.figure(figsize=(8,8), dpi=100, facecolor='w')
	ax = fig.gca(projection='3d')
	ax.plot(x_array, y_array, z_array)
#	ax.plot(drift_loc.T[0], drift_loc.T[1], drift_times)
#	ax.view_init(21,-60)
	plt.xlabel('$x$ ', size=14);	plt.ylabel('$y$ ', size=14);
	ax.set_zlabel('$z$ ', size=14)
	#plt.ticklabel_format(style='sci', axis='both', scilimits=(1,3))
	plt.suptitle('Non-adaptive time-step of Lorenz Model', fontsize=14)
	plt.show()

def make_3d_plot_from_files(adaptiveFileName1, adaptiveFileName2):
	with open(adaptiveFileName1, 'r') as f1:
		traj1 = np.recfromcsv(f1)

	with open(adaptiveFileName2, 'r') as f2:
		traj2 = np.recfromcsv(f2)
	
	fig = plt.figure(figsize=(8,8), dpi=100, facecolor='w')
	ax = fig.gca(projection='3d')
	ax.plot(traj1.x, traj1.y, traj1.z, label='$z_0 = 20.00$')
	ax.plot(traj2.x, traj2.y, traj2.z, label='$z_0 = 20.01$')
	plt.xlabel('$x$ ', size=14);	plt.ylabel('$y$ ', size=14);
	ax.set_zlabel('$z$ ', size=14)
	ax.legend()
	plt.suptitle('Lorenz model $(x,y,z)$ for $x_0=1, y_0=1$', fontsize=14)
	plt.show()

def make_2d_plot_from_files(adaptiveFileName1, adaptiveFileName2):
	with open(adaptiveFileName1, 'r') as f1:
		traj1 = np.recfromcsv(f1)

	with open(adaptiveFileName2, 'r') as f2:
		traj2 = np.recfromcsv(f2)
	
	fig = plt.figure(figsize=(8,8), dpi=100, facecolor='w')
	ax = fig.add_subplot(111)
	ax.plot(traj1.t, traj1.x, label='$z_0 = 20.00$')
	ax.plot(traj2.t, traj2.x, label='$z_0 = 20.01$')
	plt.xlabel('$t$ ', size=14);	plt.ylabel('$x(t)$ ', size=14);
	ax.legend(loc='upper left')
	plt.suptitle('Lorenz $x(t)$ for $x_0=1$, $y_0=1$', fontsize=14)
	plt.show()



def main():
	filenames = {
		'non-adaptive': 'trajectory_non_adaptive.csv',
		'adaptive-1': 'trajectory_adaptive1.csv',
		'adaptive-2': 'trajectory_adaptive2.csv'}


	# Non-adaptive trajectory.
	Trajectory_non_adaptive = Trajectory(computed_trajectory_dict=calc_trajectory_non_adaptive(dt=0.01, t0=0., tfinal=14.,
		x0=1., y0=1., z0=20.)) 
	Trajectory_non_adaptive.writeToFile(filenames['non-adaptive'])

	# Adaptive 1.
	Trajectory_adaptive_1 = Trajectory(calc_trajectory_adaptive(t0=0., tfinal=14., x0=1., y0=1.,
		z0=20.))
	Trajectory_adaptive_1.writeToFile(filenames['adaptive-1'])
	Trajectory_adaptive_1.printReport()

	# Adaptive 2. Only difference is z0=20.01
	Trajectory_adaptive_1 = Trajectory(calc_trajectory_adaptive(t0=0., tfinal=14., x0=1., y0=1.,
		z0=20.01))
	Trajectory_adaptive_1.writeToFile(filenames['adaptive-2'])
	
	make_3d_plot(Trajectory_non_adaptive.computed_trajectory_dict)
	make_3d_plot_from_files(filenames['adaptive-1'], filenames['adaptive-2'])
	make_2d_plot_from_files(filenames['adaptive-1'], filenames['adaptive-2'])


main()
