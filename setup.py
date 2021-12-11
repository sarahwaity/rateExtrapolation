from setuptools import setup, find_packages

setup(
	name = 'rateExtrapolation',
	version = '1.0.1',
	description = 'Package returns estimations for model rates.',
	long_description = 'This package simplifies reate determination from experimental data using SBstoat and Tellurium.',
	author = 'Sarah Wait',
	author_email = 'swait@uw.edu',
	url = 'https://github.com/sarahwaity/rateExtrapolation.git',
	keywords = 'tellurium rate extrapolation',
	license = 'MIT',
	install_requires = ['matplotlib == 3.3.4','numpy == 1.19.3','pandas == 1.2.4','SBstoat == 1.161', 'tellurium == 2.2.0'],
	packages = find_packages(exclude = ('docs', 'tests*', 'testing*')))
