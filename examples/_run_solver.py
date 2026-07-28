"""Run the ex03 solver with JAX_ENABLE_X64 set."""
import os, sys
os.environ['JAX_ENABLE_X64'] = 'True'

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(script_dir))

import runpy
runpy.run_path(os.path.join(script_dir, 'ex03_optimized_v3.py'), run_name='__main__')
