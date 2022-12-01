import os
cwd = os.path.abspath(os.path.dirname(__file__))
from live_wdd.prepare_resize_data import setup_data
import shutil
import subprocess

 
if __name__ == '__main__':

    type_increase = 'scan'
    type_eval = 'Memory'
    MC, path_store, list_dim, set_scan_det = setup_data(type_increase, type_eval)

    # Choose Solver
    solver = 'pyptychostem'
    formt = '_check_.json' 
        
    # Path file
    parfile ='/Users/bangun/pyptychostem/parameters.txt'
    total_result = []
    for idx in range(len(list_dim)):
        print('Processing dimension ', str(list_dim[idx]))
        path_data, path_json = set_scan_det(list_dim[idx],parfile)

        # Load Data and run reconstruction
        parfile_new = '/Local/erc-1/bangun/LiveWDD_Data/parameters.txt'
        
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
        
