#!/usr/bin/env python3
"""
Jupyter Notebook Launcher for Fraud Detection ML Project
========================================================
This script launches Jupyter notebook with the ML environment activated
"""

import subprocess
import sys
import os
from pathlib import Path

def launch_jupyter():
    """Launch Jupyter notebook with the ML environment"""
    
    # Set the working directory
    project_dir = Path("/Users/debabratapattnayak/web-dev/learnathon")
    os.chdir(project_dir)
    
    # Environment path
    env_path = project_dir / "ml_fraud_env" / "bin" / "activate"
    
    print("🚀 Launching Jupyter Notebook for Fraud Detection ML Project")
    print("=" * 60)
    print(f"📁 Working Directory: {project_dir}")
    print(f"🐍 Virtual Environment: ml_fraud_env")
    print(f"📊 Processed Data Available: ✓")
    print(f"📄 Reports Generated: ✓")
    print("=" * 60)
    
    # Launch command
    launch_cmd = f"""
    cd {project_dir} && 
    source {env_path} && 
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    """
    
    try:
        # Execute the command
        subprocess.run(launch_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Jupyter: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Jupyter notebook stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    launch_jupyter()
