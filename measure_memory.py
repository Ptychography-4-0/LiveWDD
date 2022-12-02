
import os
cwd = os.path.abspath(os.path.dirname(__file__))
path_current = os.path.abspath(os.getcwd())
from live_wdd.prepare_resize_data import setup_data
import shutil
import subprocess


# This script is used to measure the memory allocation for each algorithm
# Note that we have to pass the idp.npy for different dimensions
# Here we store it in parfile_new, the result is generated in directory Result
if __name__ == '__main__':
    
    # What dimension we want to increase
    type_increase = 'detector' # detector or scan
    # Here we fix memory evaluation
    type_eval = 'Memory'
    MC, path_store, list_dim, set_scan_det = setup_data(type_increase, type_eval)
    
    # Create directory
    os.makedirs(path_store, exist_ok = True)
    # Choose Solver
    solver = 'pyptychostem' # livewdd or pyptychostem
    formt = '.json' 
        
    # Path file for original PyPtychoSTEM Dataset
    # Here we have graphene (65,64,256,256)
    parfile ='/Users/bangun/pyptychostem-master/parameters.txt'
    total_result = []
    for idx in range(len(list_dim)):
        
        # Function to change the dataset w.r.t dimension for 
        # increasing scanning and detector dimension
        print('Processing dimension ', str(list_dim[idx]))
        path_data, path_json = set_scan_det(list_dim[idx],parfile)

        # Load parameters data after resize to new dimension and run reconstruction
        parfile_new = os.path.join(path_current,'LiveWDD_Data/parameters_new.txt')
      
        # file name
        fname = os.path.join(path_store, 'dim_idx_' + str(idx) + solver + formt)
        # Generate datasets

        print(os.getcwd())

        py = shutil.which('python3')
        command = [py, os.path.join(cwd, 'memtree.py'), '--save-json', fname, '--', py, os.path.join(cwd, 'main_memory.py'),
                        '--solver', solver, '--parfile-new', parfile_new , '--path-json', path_json,
                        '--path-data', path_data, '--mc', str(MC)]
        
        print(f"running {command} in {cwd} (from {__file__})")

        # Run Measurements
        subprocess.run(command, check=True, cwd=cwd)
        
