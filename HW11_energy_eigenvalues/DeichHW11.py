import numpy as np
import matplotlib.pyplot as plt
import sys

pc = {
	'a': 20 * 5.29e-11, # 20 Bohr radii (m).
	'h_bar': 1.05e-34, # (J s)
	'm': 9.11e-31, # mass of electron. (kg)
	'V': -13.6 * 1.6e-19 # (eV * J/eV)
}

# the function f(E) whose zeroes are the energy eigenvalues of the finite well.
def even_energy_state(E):
	return np.sqrt(E - pc['V']) * np.tan((pc['a']/pc['h_bar']) * np.sqrt(2 
		* pc['m'] * (E - pc['V']))) - np.sqrt(-E) 

def odd_energy_state(E):
	return np.sqrt(E - pc['V']) * -1/np.tan((pc['a']/pc['h_bar']) * np.sqrt(2 
		* pc['m'] * (E - pc['V']))) - np.sqrt(-E) 

# where argument of tan() is 2pi. 
def get_dist_between_eigenstate_energies():
	return  pc['V'] + ((2 * np.pi * (pc['h_bar']/pc['a']))**2) / (2 * pc['m'])

# return a centered derivative.
def approximate_derivative(f, x):
	dx = x * 1e-5 #sys.float_info.epsilon * 1e10
	return (f(x + dx/2.) - f(x - dx/2.))/dx

# Newton's method, 1-dimensional.
def find_nearest_zero(function, initial_guess, steps):
	E_n = initial_guess
	for i in range(steps):
#		print 'f(E_n): {}, E_n:{}, f\'(E_n):{}'.format(function(E_n), E_n, approximate_derivative(function, E_n)) 
		E_n = E_n - function(E_n)/approximate_derivative(function, E_n)
	return E_n



def make_plot():

	E_array = np.linspace(pc['V'], 0, 20000)
	energy_spacings_array = np.arange(pc['V'], 0, get_dist_between_eigenstate_energies())
	starting_guesses = np.linspace(pc['V']*(.91), 0, 7)
	even_zeroes = [find_nearest_zero(even_energy_state, E, 20) for E in starting_guesses]
	odd_zeroes =  [find_nearest_zero(odd_energy_state, E, 20) for E in starting_guesses]
	print 'even_zeroes: {}'.format(even_zeroes)
	print 'odd_zeroes: {}'.format(odd_zeroes)


	fig, ax = plt.subplots()
	fig.set_size_inches(8, 6)
	ax.grid()
	ax.set_xlabel('Energy (Joules)')
	ax.set_ylabel('arbitrary units ($\sqrt{E}$)')
	ax.set_ylim([-5e-8,5e-8])
	ax.set_xlim([pc['V'],0])
	ax.plot(E_array, even_energy_state(E_array),  label='even energy zeroing func')
	ax.plot(E_array, odd_energy_state(E_array), '--', label='odd energy zeroing func')
	ax.plot(starting_guesses, [0 for i in starting_guesses],  'ro', label='starting guesses')
	ax.plot(even_zeroes, [0 for i in even_zeroes],  'bo', label='even zeroes')
	ax.plot(odd_zeroes, [0 for i in odd_zeroes],  'go', label='odd zeroes')
	plt.title('Locations of eigenstate energies for finite square well')
	plt.legend()
	plt.show()



make_plot()


