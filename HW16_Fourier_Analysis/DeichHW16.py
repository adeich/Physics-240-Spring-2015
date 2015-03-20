import numpy as np
import matplotlib.pyplot as plt


# Returns amplitudes_array and frequencies_array, which together represent the fourier trans.
def do_slow_fourier_transform(input_signal_array, t_array, dt, N): 
	

	# Number of discrete sampling frequencies = t_final / dt
	# N = int(np.round((t_array[-1] - t_array[0]) / dt))
	# dt = t_array[1] - t_array[0]

	# maximum sampling frequency (according to highest theoretically measurable).
	max_frequency = ((N - 1.0) / N ) / dt

	# that constant in e**(2 pi i j k / N)
	exponent_constant = - (2. * np.pi * 1j / N) / dt

	# amplitudes_array is complex because it represents phase space.
	amplitudes_array = np.zeros(N) * (0.0 + 0.0j) 
	frequencies_array = np.linspace(0, max_frequency, N)

	for k in range(N):
		amplitudes_array[k] = np.sum(input_signal_array * np.exp(exponent_constant * t_array * k))

	return {
		'Y': amplitudes_array,
		'f': frequencies_array,
		'P': (np.abs(amplitudes_array))**2   # Power spectrum.
	}


def generate_signal(letter, f_s, N, delta_t=1):
	# timesteps.
	j_array = np.linspace(0, N * delta_t, N)

	theta_array = (2 * np.pi * f_s * j_array * delta_t ) % (2 * np.pi)

	if letter == '(a) sawtooth':
		# Sawtooth wave.
		y_array = theta_array / (2 * np.pi)	
	elif letter == '(b) square wave':
		# square wave.
		y_array = np.zeros(len(j_array))
		for i in range(len(j_array)):
			y_array[i] = 1 if (0 < theta_array[i] and theta_array[i] < np.pi) else -1	
	elif letter == '(c) square pulse':
		# square pulse.
		y_array = np.zeros(len(j_array))
		for i in range(len(j_array)):
			y_array[i] = 1 if (0 < theta_array[i] and theta_array[i] < np.pi) else 0	
	elif letter == '(d) triangle wave':
		# triangle wave.
		y_array = np.zeros(len(j_array))
		for i in range(len(j_array)):
			if (0 < theta_array[i] and theta_array[i] < np.pi):
				y_array[i] = theta_array[i] / np.pi
			else:
				y_array[i] = (2 * np.pi - theta_array[i]) / np.pi
	else:
		raise BaseException("whoa, invalid letter! '{}'".format(letter)) 

	return y_array


def plot_2by2_grid_of_signal_examples(f_s, N, dt):

	plt.figure(facecolor='w')
	plt.suptitle('The four signal functions')

	for plot_index, function_letter in enumerate(['(a) sawtooth', 
			'(b) square wave', '(c) square pulse', '(d) triangle wave']):
		t_array = np.linspace(0, N * dt, N)	
		signal_array = generate_signal(letter=function_letter, f_s=f_s, N=N, delta_t=dt)
		plt.subplot(2, 2, plot_index + 1)
		plt.plot(t_array, signal_array, 'b-') 
		plt.title(function_letter)
		plt.xlabel('time (s)'); plt.ylabel('signal')
		plt.ylim([1.5, -1.5])
		plt.grid()
		#plt.legend(loc='top right')

	plt.tight_layout()
	plt.savefig('signal_demo.png')
	plt.show()


def plot_9by9_grid_of_FTs_for_params(function_letter, dt=1.):
	
	plt.figure(facecolor='w', figsize=(8, 10))
	#	plt.suptitle(function_letter + '\n\n\n')

	for plot_index1, f_s in enumerate([0.2, 0.2123, 0.8]):
		for plot_index2, N in enumerate([50, 512, 4096]):
			plot_index = plot_index1 * 3 + plot_index2 # gives integers 0 through 8.

			# generate the time and signal array.
			t_array = np.linspace(0, N * dt, N)	
			signal_array = generate_signal(letter=function_letter, f_s=f_s, N=N, delta_t=dt)

			# find fourier transform.
			ft_results = do_slow_fourier_transform(signal_array, t_array, dt, N)		

			plt.subplot(3, 3, plot_index + 1)
			plt.plot(ft_results['f'], ft_results['P']) 
			plt.title('f_s = {}, N = {}'.format(f_s, N))
			plt.xlabel('frequency (hz)'); plt.ylabel('Power(f)')
			plt.yscale('log')
			plt.grid()

	plt.tight_layout()
	plt.savefig('FT_of_{}.png'.format(function_letter.replace(' ', '').replace('(', '').replace(')', '')))
	plt.show()



def main():
#	plot_2by2_grid_of_signal_examples(f_s=0.8, N=100, dt=1.)

	def gen_all_plots_ever():
		for function in ['(a) sawtooth', 
			'(b) square wave', '(c) square pulse', '(d) triangle wave']:
			plot_9by9_grid_of_FTs_for_params(function)	

	gen_all_plots_ever()
	
main()
