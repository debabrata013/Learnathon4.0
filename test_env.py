#!/usr/bin/env python3
"""
Test script to verify the environment works properly
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import pandas as pd
    print(f"✅ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas import error: {e}")

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import error: {e}")

try:
    import xgboost as xgb
    print(f"✅ XGBoost version: {xgb.__version__}")
except ImportError as e:
    print(f"❌ XGBoost import error: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib import error: {e}")

try:
    import seaborn as sns
    print(f"✅ Seaborn version: {sns.__version__}")
except ImportError as e:
    print(f"❌ Seaborn import error: {e}")

print("\n🎉 Environment test completed!")
