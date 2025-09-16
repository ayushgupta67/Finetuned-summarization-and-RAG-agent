# run_app.py (launcher for Streamlit)
import os, sys, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))      # ...\ai-agent-prototype
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                            # make 'src' importable

# Execute src/app.py as if it were the main script
runpy.run_module("src.app", run_name="__main__")
