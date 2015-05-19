import numpy as np
import os
import re
import argparse
import process_sound
import names_generator
import machinelearning

filename_dict = {
	'sound_samples': '../sound_samples',
	'intermediate_results': 'intermediate_results',
	'questionable_sounds': 'questionable_sounds'
}
filename_dict['pickle_filename'] = os.path.join(filename_dict['intermediate_results'],
	'trained_pickle')


def get_all_wav_files(source_directory):
	list_of_files = []
	for filename in os.listdir(source_directory):
		if os.path.splitext(filename)[1].lower() == '.wav':
			list_of_files.append(os.path.join(source_directory, filename)) 
	return list_of_files

	
# High-level function which reads in all sound files, 
# finds their peaks and generates meta-data. A sound file's
# instrument classification is read from the filename e.g.
# 'piano_c3_1.wav' gives a classification of 'piano'
def analyze_all_sound_samples(source_directory=None, list_of_files=None):

	# if file list is not specified, collect all .wav files in source_directory.
	if not list_of_files:
		if not source_directory:
			raise BaseException("needs a source directory!")
		list_of_files = get_all_wav_files(source_directory)
		
	# analyze each sound file.
	for file_number, sound_file in enumerate(list_of_files):
		print('Processing file {}/{} ...'.format(file_number, len(list_of_files)))	
		process_sound.analyze_one_sound_file(sound_file, names_generator.names_generator,
			destination_dir=filename_dict['intermediate_results'])

	print('done.')



# Read spectrum data files and return data formatted for classification.
def prepare_classification_data(source_directory, intermediate_directory):
	
	# get list of original wav files.	
	list_of_wav_files = get_all_wav_files(source_directory)
	data_list = []
	target_list = []
	

	# for each original wav file, load its normalized spectrum.
	for wav_file in list_of_wav_files:
		names_for_this_file = names_generator.names_generator(wav_file, 
			destination_directory=filename_dict['intermediate_results'])
		normalized_spectrum_name = names_for_this_file['normalized_spectrum']
		classification_string = names_for_this_file['classification']

		if len(np.loadtxt(normalized_spectrum_name)) == 5000:
			data_list.append(np.loadtxt(normalized_spectrum_name))
			target_list.append(classification_string)
		else:
			print("File '{}' is of length {}!".format(normalized_spectrum_name,
				len(np.loadtxt(normalized_spectrum_name))))

	print('set of classifications: {}'.format(set(target_list)))

	return {'data': np.array(data_list),
		'target': np.array(target_list) 
	}	
		




if __name__ == "__main__":
  # Get commandline arguments.
	parser = argparse.ArgumentParser(description='description')
	parser.add_argument('-process', action='store_true',
		help='process all wav files.')
	parser.add_argument('-mltrain', action='store_true', 
		help='train the machine learning.')
	parser.add_argument('-classify', type=str,
		help='compare 1 specified sound file.')
	args = parser.parse_args()

	if args.process:
		analyze_all_sound_samples(source_directory=filename_dict['sound_samples'])	
	elif args.mltrain:
		classification_data = prepare_classification_data(
		source_directory=filename_dict['sound_samples'],
		intermediate_directory=filename_dict['intermediate_results'])
		machinelearning.compile_classification(pickle_filename=filename_dict['pickle_filename'],
			data_array=classification_data['data'],
			target_array=classification_data['target']) 
	elif args.classify:
		process_sound.analyze_one_sound_file(sound_filename=args.classify,
			names_generator_function=names_generator.names_generator,
			destination_dir=filename_dict['questionable_sounds'])
		names = names_generator.names_generator(original_wav_filename=args.classify,
			destination_directory=filename_dict['questionable_sounds'])
		print(machinelearning.classify_from_pickle(
			pickled_filepath=filename_dict['pickle_filename'],
			normalized_spectrum_path=names['normalized_spectrum']))
	
		
