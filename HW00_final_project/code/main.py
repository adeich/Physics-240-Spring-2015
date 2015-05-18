import os
import re
import argparse
import process_sound
import names_generator

filename_dict = {
	'pickle_filename': 'hello.txt'
}


# High-level function which reads in all sound files, 
# finds their peaks and generates meta-data. A sound file's
# instrument classification is read from the filename e.g.
# 'piano_c3_1.wav' gives a classification of 'piano'
def analyze_all_sound_samples(source_directory=None, list_of_files=None):

	# if file list is not specified, collect all .wav files in source_directory.
	if not list_of_files:
		if not source_directory:
			raise BaseException("needs a source directory!")
		list_of_files = []
		for filename in os.listdir(source_directory):
			if os.path.splitext(filename)[1].lower() == '.wav':
				list_of_files.append(os.path.join(source_directory, filename)) 
			
	# analyze each sound file.
	for file_number, sound_file in enumerate(list_of_files):
		print('Processing file {}/{} ...'.format(file_number, len(list_of_files)))	
		process_sound.analyze_one_sound_file(sound_file, names_generator.names_generator,
			destination_dir='intermediate_results')

	print('done.')


def prepare_machine_learning(source_directory):
	pass



def main():
	pass


if __name__ == "__main__":
  # Get commandline arguments.
	parser = argparse.ArgumentParser(description='description')
	parser.add_argument('-process', action='store_true',
                  help='process all wav files.')
	parser.add_argument('-mltrain', action='store_true',
                  help='train the machine learning.')
	args = parser.parse_args()

	if args.process:
		analyze_all_sound_samples(source_directory='../sound_samples')	
	if args.mltrain:
		prepare_machine_learning()
