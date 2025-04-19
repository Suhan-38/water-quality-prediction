import sys
import importlib

# Print Python version
print(f"Python version: {sys.version}")

# Check NumPy
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy path: {np.__path__}")
    # Check if _core exists
    try:
        from numpy import _core
        print("numpy._core is available")
    except ImportError:
        print("numpy._core is NOT available")
except ImportError as e:
    print(f"NumPy import error: {e}")

# Check scikit-learn
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"scikit-learn import error: {e}")

# Check joblib
try:
    import joblib
    print(f"joblib version: {joblib.__version__}")
except ImportError as e:
    print(f"joblib import error: {e}")

# Check pandas
try:
    import pandas as pd
    print(f"pandas version: {pd.__version__}")
except ImportError as e:
    print(f"pandas import error: {e}")
