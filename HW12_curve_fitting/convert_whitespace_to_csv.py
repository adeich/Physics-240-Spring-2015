import re

def make_new_file(sInputF, sOutputF=None):
	if not sOutputF:
		sOutputF = 'CommaData.csv'
	with open(sInputF, 'r') as sInputF:
		with open(sOutputF, 'w') as sOutputF:
			for line in sInputF:
				# replace each section of whitespace with comma.
				new_CSV_line = ','.join(re.findall('\"[^\"]*\"|\S+', line))
				sOutputF.write(new_CSV_line)
				sOutputF.write('\n')



make_new_file('GlobalAnnualMeanSurfaceAirTemperatureChange.csv')
