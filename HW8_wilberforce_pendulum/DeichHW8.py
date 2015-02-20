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
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(trajectory['t_array'], trajectory['z'], label='$z$')
	ax.plot(trajectory['t_array'], trajectory['theta'], label='$\theta$')
	ax.legend(loc='upper right')
	plt.show()


def main():
	trajectory = generate_trajectory(dt = 0.01,
		timesteps=1e4,
		z0=0.10, 		# m
		theta0=0., 	# rad
		m=0.5,			# kg
		I=1.0e-3, 	# kg m^2
		k=5, 				# N/m
		eps=10e-2, 	# N
		delta=10e-3 # N m 
	)
	make_plot(trajectory)


main()
	
