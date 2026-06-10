"""
Installation and Testing Script
Run this to verify the project setup and installation
"""

import subprocess
import sys
import os

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("[OK] Python 3.8+ detected")
        return True
    else:
        print("[ERROR] Python 3.8+ required")
        return False


def install_requirements():
    """Install required packages."""
    print("\n" + "="*70)
    print("Installing dependencies from requirements.txt...")
    print("="*70)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        print("[OK] All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        print("\nTry manual installation:")
        print("  pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit")
        return False


def test_imports():
    """Test if all required modules can be imported."""
    print("\n" + "="*70)
    print("Testing module imports...")
    print("="*70)
    
    modules = {
        'pandas': 'Data processing',
        'numpy': 'Numerical computation',
        'sklearn': 'Machine learning',
        'xgboost': 'Gradient boosting',
        'shap': 'Model explainability',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical plots',
        'streamlit': 'Web framework'
    }
    
    all_ok = True
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"[OK] {module:15s} - {description}")
        except ImportError:
            print(f"[FAIL] {module:15s} - {description}")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Verify project directory structure."""
    print("\n" + "="*70)
    print("Checking project structure...")
    print("="*70)
    
    required_dirs = ['src', 'models', 'outputs', 'reports', 'data', 'app']
    required_files = ['train_pipeline.py', 'requirements.txt', 'README.md']
    
    all_ok = True
    
    print("Directories:")
    for dir_name in required_dirs:
        exists = os.path.isdir(dir_name)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {dir_name}/")
        if not exists:
            all_ok = False
    
    print("\nFiles:")
    for file_name in required_files:
        exists = os.path.isfile(file_name)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {file_name}")
        if not exists:
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "CUSTOMER CHURN ML PROJECT SETUP" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Install Dependencies", install_requirements),
        ("Module Imports", test_imports)
    ]
    
    results = []
    for check_name, check_func in checks:
        print()
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"[ERROR] {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    for check_name, result in results:
        status = "[OK]" if result else "[FAILED]"
        print(f"{status} {check_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("SUCCESS! Project is ready to use.")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run the training pipeline:")
        print("     python train_pipeline.py")
        print("\n  2. Launch the Streamlit app:")
        print("     streamlit run app/streamlit_app.py")
        print("\n  3. Read the documentation:")
        print("     - README.md (complete guide)")
        print("     - project_report.md (technical details)")
        print("     - QUICKSTART.md (quick reference)")
        return 0
    else:
        print("\n" + "="*70)
        print("SETUP INCOMPLETE")
        print("="*70)
        print("\nPlease fix the issues above and try again.")
        print("\nFor manual installation:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
