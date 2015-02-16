import numpy as np
import matplotlib.pyplot as plt

# global physical parameters.
pc = {
	'G': 6.67e-11, # m^3/kg/s^2,
	'M_sun': 1.99e30, # kg
	'M_earth': 5.9e24, # kg
	'r_earth_sun': 1.50e11, # m 
	'v_earth': 2.98e4,  # m/s
	'year_in_seconds': 365. * 24 * 3600 #s
	}	

# Orbit integration for one orbital period.
# This function is called by several of the HW parts for
# varying purposes.
def compute_1_orbit(dt, r0, v0):

	r_list = [r0]
	v_list = [v0]

	def orbit_has_just_passed_positive_x_axis(r):
		return ((len(r) > 3) and 
			(r[-1][1] * r[-2][1] < 1) and # y-value has just switched signs.
			(r[-1][1] > 0)) # y-value positive

	# Euler-Cromer
	while not orbit_has_just_passed_positive_x_axis(r_list):
		# Acceleration given by a = F/m = ( GM / r^2 ) * r_hat = (GM / r^2 ) * r_vec
		currA = - pc['G'] * pc['M_sun'] * r_list[-1] / np.linalg.norm(r_list[-1])**3
		v_list.append(v_list[-1] + currA * dt)
		r_list.append(r_list[-1] + v_list[-1] * dt)

	return {'r_array': np.array(r_list),
		'v_array': np.array(v_list),
		't_array': np.arange(0, dt*len(r_list), dt)}
		



def do_part1a():
	output_dict = compute_1_orbit(pc['year_in_seconds']/1e3, np.array([pc['r_earth_sun'], 0]),
		np.array([0, pc['v_earth']]))

	print '{} steps. {} seconds.'.format(len(output_dict['t_array']), output_dict['t_array'][-1])


def do_part1b():
	pass

def do_part1c():
	pass

def do_part2a():
	pass

def do_part2b():
	pass




def do_all_of_HW7():
	do_part1a()
	do_part1b()
	do_part1c()
	do_part2a()
	do_part2b()


do_all_of_HW7()
