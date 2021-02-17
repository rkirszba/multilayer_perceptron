import subprocess
import os
import sys
import re
import pandas as pd

NB_TESTS = 100
FLOAT_REGEX = r'^Binary cross entropy cost = (\d+\.?\d+)$'

if __name__ == '__main__':

	perfs = []

	try:
		for test in range(NB_TESTS):
			print('Test {}'.format(test))
			print('    Creating dataset...')
			subprocess.call(['python', 'evaluation.py'], stdout=subprocess.PIPE)
			print('    Training model...')
			subprocess.call(['python', 'train.py', 'data_training.csv'], stdout=subprocess.PIPE)
			print('    Predicting...')
			output = subprocess.check_output(['python', 'predict.py', 'data_test.csv']).decode('utf-8')
			match = re.search(FLOAT_REGEX, output, flags=re.MULTILINE)
			perfs.append(float(match.group(1)))

		series = pd.Series(perfs)
		under_08 = series.groupby(series < 0.08).count()[True]
		print('--------------------------')
		print('Series:')
		print(series.to_string())
		print()
		print('Mean = {}'.format(series.mean()))
		print()
		print('Under 0.08 = {}/{} ({:.2f}%)'.format(under_08, NB_TESTS, (under_08 / NB_TESTS) * 100))
		print('--------------------------')

	except Exception as e:
		print('Something went wrong: {}'.format(e))