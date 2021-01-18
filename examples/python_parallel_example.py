from joblib import Parallel, delayed
import multiprocessing

inputs = range(1,3815)  ## formulate your inputs in some form that you can call them with an index. I.e. for image processing using PIMS load frames with frames[i] in the function your calling

def mosCoordinate(i):
    ####your code here

num_cores = multiprocessing.cpu_count()
    
results = Parallel(n_jobs=num_cores)(delayed(mosCoordinate)(i) for i in inputs)