import numpy as np
import os
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import findpeaks
import names_generator


# Reads the wav file. Returns a 1-d array of signal data. 
def read_wav_file(filename):
	wavfile_data = scipy.io.wavfile.read(filename)
	return {'signal_array': wavfile_data[1].T[0], 'sample_rate': wavfile_data[0]}


def calc_fft(signal_data, timestep):
	power_spectrum = np.fft.fft(signal_data)
	index_of_largest_positive = np.floor(len(power_spectrum) / 2)
	positive_half_of_power_spectrum = power_spectrum[0: index_of_largest_positive]
	all_frequencies = np.fft.fftfreq(n=len(signal_data), d=timestep)
	positive_half_of_frequencies = all_frequencies[0: index_of_largest_positive]
	return {'fft_array': np.abs(positive_half_of_power_spectrum),
		'freqs_array': positive_half_of_frequencies}


#def find_xy_values_of_peaks(data_array, frequencies_array):
#	peak_indices = signal.find_peaks_cwt(data_array, widths=np.arange(5, 20), min_snr=2,
#		 noise_perc=0.1)
#	peak_amplitudes= []
#	peak_freqs = []
#	for peak_index in peak_indices:
#		peak_amplitudes.append(data_array[peak_index])
#		peak_freqs.append(frequencies_array[peak_index])
#	print peak_freqs
#	return {'freq': np.array(peak_freqs), 'amplitude': np.array(peak_amplitudes)}
	
# should return dict of indices of peaks, (x, y) array.
def get_peak_data(x, y):
	peak_data = findpeaks.find_peaks(x=x, y=np.abs(y), SlopeThreshold=200.,
		AmpThreshold=0., smoothwidth=25, peakgroup=9., smooth_iterations=5, polyfitdegree=37)

	return peak_data
	

# "Normalize" power spectrum by shifting the first peak to be at 100HZ and amplitude 1.
def make_normalized_power_spectrum(fft_data, peak_data):
	
	# Shift elements of array so that first peak occurs at element 'normalized_firstpeak_j'.
	def roll_and_pad(original_array, orig_first_peak_j, new_first_peak_j, total_new_length):
		if not (type(orig_first_peak_j) == int) or not (type(new_first_peak_j) == int):
			raise BaseException('type must be integer!')
		roll_amount = new_first_peak_j - orig_first_peak_j 
		rolled_intermediate_array = np.roll(original_array, roll_amount)
		if roll_amount < 1:
			rolled_and_padded_array = rolled_intermediate_array[: total_new_length]
		else:
			rolled_and_padded_array = np.lib.pad(rolled_intermediate_array[roll_amount:], (roll_amount, 0), 'constant')[: total_new_length]
		return rolled_and_padded_array

	# the amplitude of the power spectrum at each array index.
	orig_spectrum_y_array = fft_data['fft_array']

	# the frequency of the power spectrum at each array index.
	orig_spectrum_x_array = fft_data['freqs_array']

	if len(peak_data['actual_peak_xyj_list']) == 0:
		raise BaseException('No peaks!')

	firstpeak_x, firstpeak_y, firstpeak_j = peak_data['actual_peak_xyj_list'][0]
	firstpeak_j = int(firstpeak_j)

#	print 'first peak j: {}'.format(firstpeak_j)
#	print 'first peak x: {}'.format(firstpeak_x)
#	print 'first peak y: {}'.format(orig_spectrum_y_array[firstpeak_j])
#	print 'first peak y: {}'.format(firstpeak_y)

	total_length = 5000
	normalized_firstpeak_j= 100

	normalized_spectrum_x_array = roll_and_pad(original_array=orig_spectrum_x_array, 
		orig_first_peak_j=firstpeak_j, new_first_peak_j=normalized_firstpeak_j, 
		total_new_length=total_length)
	normalized_spectrum_y_array = roll_and_pad(original_array=orig_spectrum_y_array, 
		orig_first_peak_j=firstpeak_j, new_first_peak_j=normalized_firstpeak_j, 
		total_new_length=total_length) / firstpeak_y

#	print "y of normalized first peak: {}".format(normalized_spectrum_y_array[normalized_firstpeak_j])

	return {'fft_array': normalized_spectrum_y_array,
		'freqs_array': normalized_spectrum_x_array,
		'first_peak_j': normalized_firstpeak_j}
	

def plot_just_spectrum(spectrum_array, frequencies_array, timestep, save_to,
	index_of_first_peak, show_plot=False):
	
	plt.figure(facecolor='w')

	plt.subplot(1, 1, 1)
	plt.plot(frequencies_array, spectrum_array) 
	plt.title('normalized spectrum')
	plt.xlabel('frequency'); plt.ylabel('amplitude')
	plt.xlim(50, 15000)
#	plt.yscale('log'); 
	plt.xscale('log')
	plt.grid()
	plt.tight_layout()

	plt.axvline(x=frequencies_array[index_of_first_peak], color='black')

	plt.savefig(save_to)
	if show_plot:
		plt.show()


def plot_signal_and_spectra(signal_array, spectrum_array, frequencies_array, peak_data,
	 timestep, save_to, show_plot=False):
	
	spectrum_array = np.abs(spectrum_array)

	peakpointsJ = peak_data['actual_peak_xyj_list'].T[2]
	peakpointsX = peak_data['actual_peak_xyj_list'].T[0]
	peakpointsY = peak_data['actual_peak_xyj_list'].T[1]
	polyfit_y_array = peak_data['polyfit_y_array']	
	
	plt.figure(facecolor='w')

	
	plt.subplot(2, 1, 1) 
	plt.plot(np.linspace(0, len(signal_array)*timestep, len(signal_array)), signal_array)
	plt.title('signal')
	plt.xlabel('time'); plt.ylabel('signal')
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(frequencies_array, spectrum_array) 

	### plot polynomial fit
	
	plt.plot(peak_data['polyfit_x_array'], polyfit_y_array)

	### end polynomial fit.


	plt.plot(peakpointsX, peakpointsY, 'ro')
	plt.title('spectrum')
	plt.xlabel('frequency'); plt.ylabel('amplitude')
	plt.xlim(50, 15000)
	plt.yscale('log'); plt.xscale('log')
	plt.grid()
	plt.tight_layout()


	plt.savefig(save_to)
	if show_plot:
		plt.show()


def analyze_one_sound_file(sound_filename, names_generator_function, destination_dir):
	# generate associated filenames.
	names_dict = names_generator_function(sound_filename, destination_directory=destination_dir)

	# load WAV file as array.
	channel_data = read_wav_file(sound_filename)
	signal_array = channel_data['signal_array']
	timestep = 1./channel_data['sample_rate']

	# do FFT to find power spectrum.
	fft_data = calc_fft(signal_array, timestep)

	# find peaks in power spectrum.
	peak_data = get_peak_data(y=fft_data['fft_array'], 
		x=fft_data['freqs_array'])

	# make plot including signal, power spectrum, and peaks.
	plot_signal_and_spectra(signal_array=signal_array,
		spectrum_array=fft_data['fft_array'],
		frequencies_array=fft_data['freqs_array'],
		peak_data=peak_data,
		timestep=timestep,
		save_to=names_dict['raw_plot'])

	# normalize power spectrum and peaks around first peak; make same plots as above.
	normalized_fft_data = make_normalized_power_spectrum(fft_data, peak_data)
	plot_just_spectrum(
		spectrum_array=normalized_fft_data['fft_array'],
		frequencies_array=normalized_fft_data['freqs_array'],
		index_of_first_peak=normalized_fft_data['first_peak_j'],
		timestep=timestep,
		save_to=names_dict['normalized_plot'])


	# save to file: (1) raw power spectrum and (2) normalized spectrum (array -> txt). 
	np.savetxt(names_dict['raw_spectrum'], fft_data['fft_array'])
	np.savetxt(names_dict['normalized_spectrum'], normalized_fft_data['fft_array'])



if __name__ == "__main__":
  # Get commandline arguments.
	parser = argparse.ArgumentParser(description='description')
	parser.add_argument('soundclip', type=str,
                   help='sound file in wav format')
	args = parser.parse_args()

	analyze_one_sound_file(sound_filename=args.soundclip, 
		names_generator_function=names_generator.names_generator, 
		destination_dir='intermediate_results')
