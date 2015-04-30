import numpy as np
import os
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import findpeaks


# Reads the wav file. Returns a 1-d array of signal data. 
def read_wav_file(filename):
	wavfile_data = scipy.io.wavfile.read(filename)
	return {'signal_array': wavfile_data[1].T[0], 'sample_rate': wavfile_data[0]}


def calc_fft(signal_data, timestep):
	return {'fft_array': np.fft.fft(signal_data), 
		'freqs_array': np.fft.fftfreq(n=len(signal_data), d=timestep)}


def find_xy_values_of_peaks(data_array, frequencies_array):
	peak_indices = signal.find_peaks_cwt(data_array, widths=np.arange(5, 20), min_snr=2,
		 noise_perc=0.1)
	peak_amplitudes= []
	peak_freqs = []
	for peak_index in peak_indices:
		peak_amplitudes.append(data_array[peak_index])
		peak_freqs.append(frequencies_array[peak_index])
	print peak_freqs
	return {'freq': np.array(peak_freqs), 'amplitude': np.array(peak_amplitudes)}
	

def find_indices_of_peaks(data_array):
	#peak_indices = signal.find_peaks_cwt(data_array, np.linspace(len(data_array)/2., len(data_array), 2))
	peak_indices = findpeaks.find_peaks(data_array, range(len(data_array)), SlopeThreshold=0.003,
		AmpThreshold=0.5, smoothwidth=7, peakgroup=9., smoothtype=3)
	
	return peak_indices
	



def plot_signal_and_spectra(signal_array, spectrum_array, frequencies_array, peak_data,
	 timestep, save_to):
	
	spectrum_array = np.abs(spectrum_array)
	
	plt.figure(facecolor='w')

	
	plt.subplot(2, 1, 1) 
	plt.plot(np.linspace(0, len(signal_array)*timestep, len(signal_array)), signal_array)
	plt.title('signal')
	plt.xlabel('time'); plt.ylabel('signal')
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(frequencies_array, spectrum_array) 
	plt.plot(peak_data['freq'], peak_data['amplitude'], 'ro')
	plt.title('spectrum')
	plt.xlabel('frequency'); plt.ylabel('amplitude')
	plt.xlim(50, 15000)
	plt.yscale('log'); plt.xscale('log')
	plt.grid()
	plt.tight_layout()


	plt.savefig(save_to)
	plt.show()

def main(soundfile):

	# Read the audio file.
	channel1_data = read_wav_file(soundfile)
	timestep = 1./channel1_data['sample_rate']


	# Compute the FFT of the audio file. 
	fft_data = calc_fft(channel1_data['signal_array'], timestep)
	print np.shape(fft_data['freqs_array'])

	# Compute the peaks of the FFT data
	peak_data = find_xy_values_of_peaks(data_array=fft_data['fft_array'],
		 frequencies_array=fft_data['freqs_array'])	
	print peak_data['freq']
	


	# Plot the signal and the power spectrum.
	image_file_name = '{}_spectrum.png'.format(os.path.splitext(os.path.basename(soundfile))[0])
	plot_signal_and_spectra(signal_array=channel1_data['signal_array'], 
		spectrum_array=fft_data['fft_array'],
		frequencies_array=fft_data['freqs_array'], peak_data=peak_data, 
		timestep=timestep, save_to=image_file_name)
		


if __name__ == "__main__":
  # Get commandline arguments.
  parser = argparse.ArgumentParser(description='description')
  parser.add_argument('--soundclip', '-s', type=str,
                   help='sound file in wav format')
  args = parser.parse_args()
  # Call main function.
  main(soundfile=args.soundclip)




