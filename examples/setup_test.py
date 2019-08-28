import os, sys, os.path, tempfile, subprocess, shutil


filename = r'openMPtest.c'

compiler = os.getenv('CC', 'cc')


with open(os.devnull, 'w') as fnull:
        exit_code = subprocess.call([compiler, '-Xpreprocessor', '-fopenmp', '-lomp', filename], stdout=fnull, stderr=fnull)

print(exit_code)