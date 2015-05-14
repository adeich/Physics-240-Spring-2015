import os
import re

def names_generator(original_wav_filename, destination_directory=None):
	# 'piano_1_foo_bar.wav' -> 'piano_1' as sound ID. 
	# and 'piano' as classification string.


	# if destination directory is not specified, the new files are put in
	# the directory of their specified pathname. 
	if destination_directory:
		dirname = destination_directory
		if not (os.path.isdir(destination_directory)):
			os.makedirs(destination_directory)
			print('made new directory "{}"'.format(destination_directory)) 
	else:
		dirname = os.path.dirname(original_wav_filename)
	basename = os.path.basename(original_wav_filename)
	occurrences_of_underscore = [m.start() for m in re.finditer('_', basename)]
	if len(occurrences_of_underscore) < 2:
		raise BaseException('needs more underscores! "{}"'.format(basename))
	classification = basename[:occurrences_of_underscore[0]]
	soundID = basename[:occurrences_of_underscore[1]]

	return {
		'original': original_wav_filename,
		'classification': classification,
		'raw_plot' : os.path.join(dirname, '{}_rawplot.png'.format(soundID)),
		'raw_signal': os.path.join(dirname, '{}_rawsignal.txt'.format(soundID)),
		'raw_spectrum': os.path.join(dirname, '{}_rawspectrum.txt'.format(soundID)),
		'normalized_spectrum': os.path.join(dirname, '{}_normalizedspectrum.txt'.format(soundID)),
		'normalized_plot': os.path.join(dirname, '{}_normalizedplot.png'.format(soundID))
	}
	


