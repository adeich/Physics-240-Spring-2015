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

# Orbit integration for one orbital period. This function is 
# called by several of the HW parts for varying purposes.
def compute_1_orbit(dt, r0, v0):

	r_list = [r0]
	v_list = [v0]

	def orbit_has_just_passed_positive_x_axis(r):
		return ((len(r) > 3) and 
			(r[-1][1] * r[-2][1] < 1) and # y-value has just switched signs.
			(r[-1][1] > 0)) # y-value positive

	# Euler-Cromer
	while not orbit_has_just_passed_positive_x_axis(r_list):
		# Acceleration given by a = F/m = ( GM / r^2 ) * r_hat = (GM / r^3 ) * r_vec
		currA = - pc['G'] * pc['M_sun'] * r_list[-1] / np.linalg.norm(r_list[-1])**3
		v_list.append(v_list[-1] + currA * dt)
		r_list.append(r_list[-1] + v_list[-1] * dt)

	return {'r_array': np.array(r_list),
		'v_array': np.array(v_list),
		't_array': np.arange(0, dt*len(r_list), dt)}
		

# Compute eccentricity, perihelion, and semi-major axis and compare them to 
# analytic values.
def do_part1():

	def study_orbit(r0_vec, v0_vec):

		measurements= {}

		# Perform euler-cromer integration to find r and v for all timesteps in an orbit.
		orbit_results = compute_1_orbit(pc['year_in_seconds']/1e3, r0_vec, v0_vec)

		measurements['period'] = orbit_results['t_array'][-1]
	
		# find the perihelion and aphelion (biggest and smallest orbital radius)
		max_r_mag = 0
		min_r_mag = np.inf
		for r_vec in orbit_results['r_array']:
			magnitude = np.linalg.norm(r_vec)
			if magnitude > max_r_mag: 
				max_r_mag = magnitude
			if magnitude < min_r_mag:
				min_r_mag = magnitude
		measurements['r_ap'] = r_ap = max_r_mag
		measurements['r_per'] = r_per = min_r_mag 

		# Compute eccentricity using perihelion and aphelion.
		measurements['e_empiric'] = (r_ap - r_per) / (r_ap + r_per)
		measurements['a_empiric'] = (r_per + r_ap) / 2.

		### Compute eccentricity using specific energy and angular momentum.

		# Reduced mass.
		mu = (pc['M_sun'] + pc['M_earth']) / (pc['M_sun'] + pc['M_earth'])

		# Specific energy E/m = - GM/r
		measurements['E_spec'] = E_spec = - pc['G'] * (pc['M_earth'] + pc['M_sun']) / (2 * measurements['a_empiric'])

		# Specific angular momentum L = r cross v / mu
		measurements['L_spec'] = L_spec = np.cross(orbit_results['r_array'][0], orbit_results['v_array'][0]) / mu

		# Eccentricity, computed analytically.
		measurements['e_analytic'] = np.sqrt(1 + (2 * E_spec * L_spec**2)/(pc['G']**2 * pc['M_sun']**2))		

		# From T**2 / r**3 = 4pi**2 / GM
		true_value = (4 * np.pi**2) / (pc['G'] * pc['M_sun'])
		measured_value = measurements['period']**2 / measurements['a_empiric']**3 
		measurements['KIIIerror'] = (true_value - measured_value) / true_value

		output_lines = ['r0: {}; v0: {}'.format(r0_vec, v0_vec)]
		for key, value in measurements.iteritems():
			output_lines.append('\t{}: {:.2e}'.format(key, value))

		print '\n'.join(output_lines)
		

	print 'Part 1a: eccentricity and semi-major axis.'
	
	# Near-circular orbit. These are just Earth orbit parameters.
	study_orbit(r0_vec=np.array([pc['r_earth_sun'], 0]), v0_vec=np.array([0, pc['v_earth']]))

	# slightly more eccentric orbit.
	study_orbit(r0_vec=np.array([pc['r_earth_sun'], 0]), v0_vec=np.array([0, pc['v_earth'] * 0.8 ]))

	
def do_part2a():
	pass

def do_part2b():
	pass




def do_all_of_HW7():
	do_part1()
	do_part2a()
	do_part2b()


do_all_of_HW7()
