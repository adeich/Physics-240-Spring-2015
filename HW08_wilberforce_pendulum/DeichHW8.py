import numpy as np
import matplotlib.pyplot as plt

# Calculate the derivative specific to the Wilburforce Pendulum.
def calc_dx_dy(x, y, c1, c2):
	return - (c1 * x) - (c2 * y / 2)
	

# RK4: at a given x and y (i.e. z and theta or vice versa), return the next x. 
def calc_RK4(dt, x, y, c1, c2):
	k1 = calc_dx_dy(x, y, c1, c2)
	k2 = calc_dx_dy(x + k1 * dt/2., y, c1, c2)
	k3 = calc_dx_dy(x + k2 * dt/2., y, c1, c2)
	k4 = calc_dx_dy(x + k3 * dt, y, c1, c2)
	return (k1 + 2. * (k2 + k3) + k4) * dt/6.

def generate_trajectory(dt, timesteps, z0, theta0, m, I, k, eps, delta):
	timesteps = int(timesteps)
	z = [z0]
	theta = [theta0]
	z_dot = [0.]
	theta_dot = [0.]
	t_array = np.linspace(0, dt * timesteps, timesteps)	
	for i in range(1, len(t_array)):
		z_dot.append(z_dot[-1] + calc_RK4(dt, z[-1], theta[-1], k/m, eps/m))
		theta_dot.append(theta_dot[-1] + calc_RK4(dt, theta[-1], z[-1], delta/I, eps/I))
		z.append(z[-1] + z_dot[-1] * dt)
		theta.append(theta[-1] + theta_dot[-1] * dt)
	return {'z': np.array(z), 'theta': np.array(theta), 't_array': t_array}
	

def make_plot(trajectory):
	fig, ax1 = plt.subplots()
	
#	plt.rc('text', usetex=True)

	ax1.grid()
	ax1.set_xlabel('time (s)')
	ax1.set_ylabel('z (m)', color='b')
	ax1.set_ylim([-.15, .15])
	ax1.plot(trajectory['t_array'], trajectory['z'], 'b-',
		 label='$z_0$ = {}m'.format(trajectory['z'][0]))
	for tl in ax1.get_yticklabels():
		tl.set_color('b')

	ax2 = ax1.twinx()
	ax2.set_ylim([- np.pi,  np.pi])
	ax2.plot(trajectory['t_array'], trajectory['theta'], 
		'g-', label='$theta_0$ = {}pi'.format(trajectory['theta'][0]/np.pi)) 
	ax2.set_ylabel('$theta$ (rad)', color='g')
	for tl in ax2.get_yticklabels():
		tl.set_color('g')
	ax1.legend(loc='upper left')
	ax2.legend(loc='upper right')
	plt.show()


def main():
	trajectory1 = generate_trajectory(dt = 0.01,
		timesteps=9e3,
		z0=0.10,	# m
		theta0=0., 	# rad
		m=0.5,	# kg
		I=10.0e-4,	# kg m^2
		k=5, 	# N/m
		eps=1e-2, # N
		delta=10e-3 # N m 
	)
	make_plot(trajectory1)


	trajectory2 = generate_trajectory(dt = 0.01,
		timesteps=9e3,
		z0=0.,	# m
		theta0=np.pi, 	# rad
		m=0.5,	# kg
		I=10.0e-4,	# kg m^2
		k=5, 	# N/m
		eps=1e-2, # N
		delta=10e-3 # N m 
	)

	
	make_plot(trajectory2)


main()
	
