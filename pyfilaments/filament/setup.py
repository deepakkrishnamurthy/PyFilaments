import numpy
import os, sys, os.path, tempfile, subprocess, shutil
from sys import platform
from distutils.core import setup
# from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

def check_for_openmp():
	##Adapted from Goldbaum reply. See https://github.com/pynbody/pynbody/issues/124
	# Create a temporary directory
	tmpdir = tempfile.mkdtemp()
	curdir = os.getcwd()
	os.chdir(tmpdir)

	# Get compiler invocation
	compiler = os.getenv('CC', 'cc')

	# Attempt to compile a test script.
	# See http://openmp.org/wp/openmp-compilers/
	filename = r'test.c'
	# filename = r'openMPtest.c')
	file = open(filename,'w')
	file.write(
		"#include <omp.h>\n"
		"#include <stdio.h>\n"
		"int main() {\n"
		"#pragma omp parallel\n"
		"printf(\"Hello from thread %d, nthreads %d\\n\", omp_get_thread_num(), omp_get_num_threads());\n"
		"}")

	# Fixed major bug. File needs to be closed before compilation can commence.
	file.close()

	# Check which platform to define the OpenMP flags
	if platform == "linux" or platform == "linux2":
		print("linux system")
		with open(os.devnull, 'w') as fnull:
			exit_code = subprocess.call([compiler, '-fopenmp', filename], stdout=fnull, stderr=fnull)
		

	elif platform == 'darwin':
		print("OSX system")
		with open(os.devnull, 'w') as fnull:
			exit_code = subprocess.call([compiler, '-Xpreprocessor', '-fopenmp', '-lomp', filename], stdout=fnull, stderr=fnull)
		

	elif platform == "win32":
		print("Windows")
		with open(os.devnull, 'w') as fnull:
			exit_code = subprocess.call([compiler, '-Xpreprocessor', '-fopenmp', '-lomp', filename], stdout=fnull, stderr=fnull)


	
		# exit_code = subprocess.call([compiler, '-Xpreprocessor', '-fopenmp', '-lomp', filename], stdout=fnull, stderr=fnull)

	# Clean up
	os.chdir(curdir)
	shutil.rmtree(tmpdir)

	if exit_code == 0:
		print(exit_code)
		print('OpenMP found!')
		return True
	else:
		print(exit_code)
		print('No OpenMP!')
		return False

if check_for_openmp() == True:
	# omp_args = ['-Xpreprocessor', '-fopenmp', '-lomp']
	if platform == "linux" or platform == "linux2":
		print("linux system")
		omp_args = ['-fopenmp']
	elif platform == 'darwin':
		print("OSX system")
		omp_args = ['-Xpreprocessor', '-fopenmp', '-lomp']
	elif platform == "win32":
		print("Windows")
		omp_args = ['-Xpreprocessor', '-fopenmp', '-lomp']
		
	print('OpenMP found!')
else:
	omp_args = None
	print('No OpenMP found!')

setup(
	name='filament',
	version='1.0.0',
	url='https://github.com/deepakkrishnamurthy/PyFilaments',
	author='Deepak Krishnamurthy',
	author_email='deepak90@stanford.edu',
	license='MIT',
	description='python library for computing active filament dynamics in Stokes flows',
	long_description='pyfilaments is a library for computing active filament dynamic using active colloid hydrodynamics. It is built on top of pystokes.',
	platforms='tested on LINUX',
	ext_modules=cythonize([ Extension("filament/*", ["filament/*.pyx"],
		include_dirs=[numpy.get_include()],
		extra_compile_args=omp_args,
		extra_link_args=omp_args 
		)]),
	libraries=[],
	packages=['filament'],
	package_data={'filament': ['*.pxd']}
)

