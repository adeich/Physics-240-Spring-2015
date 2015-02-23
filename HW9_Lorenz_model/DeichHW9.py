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
# 'derivative_func' vector-function's derivative, and 'pos_vec', 
# is the current position vector.
def calc_RK4_vec(derivative_func, pos_vec, dt):
	k1 = derivative_func(pos_vec)
	k2 = derivative_func(pos_vec + k1 * dt/2.)
	k3 = derivative_func(pos_vec + k2 * dt/2.)
	k4 = derivative_func(pos_vec + k3 * dt)
	return pos_vec + (k1 + 2. * (k2 + k3) + k4) * dt/6.

# same form as 'rka()' from code examples.
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

#		next_delta_t = S1 * curr_delta_t * np.abs(max_allowable_err / curr_error_estimate)**0.2
#		next_delta_t = np.min(next_delta_t, S2 * curr_delta_t)
#		next_delta_t = np.max(next_delta_t, curr_delta_t / S2)
#		curr_delta_t = next_delta_t

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
	plt.suptitle('hello', fontsize=14)
	plt.show()


def main():
	trajectory_dict_non_adaptive = calc_trajectory_non_adaptive(dt=0.01, t0=0., tfinal=14.,
		x0=1., y0=1., z0=20.) 

	trajectory_dict_adaptive = calc_trajectory_adaptive(t0=0., tfinal=14., x0=1., y0=1.,
		z0=20.)

	# for debugging.
	np.savetxt('pos_array.csv', trajectory_dict_non_adaptive['pos_array'], delimiter=',')

	make_3d_plot(trajectory_dict_non_adaptive)
	make_3d_plot(trajectory_dict_adaptive)

main()
