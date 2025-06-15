#!/usr/bin/env python3
"""
Build and test script for MultiPromptify package
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"Command: {cmd}")
    print("="*50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False

def main():
    """Main build and test process."""
    print("🚀 MultiPromptify Package Build & Test")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("setup.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Clean previous builds
    print("\n1. Cleaning previous builds...")
    for path in ["build", "dist", "*.egg-info"]:
        if Path(path).exists():
            run_command(f"rm -rf {path}", f"Removing {path}")
    
    # Step 2: Install build dependencies
    print("\n2. Installing build dependencies...")
    if not run_command("pip install build twine wheel", "Installing build tools"):
        print("❌ Failed to install build dependencies")
        return False
    
    # Step 3: Build the package
    print("\n3. Building the package...")
    if not run_command("python -m build", "Building wheel and source distribution"):
        print("❌ Failed to build package")
        return False
    
    # Step 4: Check the package
    print("\n4. Checking the package...")
    if not run_command("twine check dist/*", "Validating package"):
        print("❌ Package validation failed")
        return False
    
    # Step 5: Test local installation
    print("\n5. Testing local installation...")
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("❌ Failed to install package locally")
        return False
    
    # Step 6: Test import
    print("\n6. Testing import...")
    try:
        from multipromptify import MultiPromptifyAPI
        print("✅ Successfully imported MultiPromptifyAPI!")
        
        # Quick test
        mp = MultiPromptifyAPI()
        print("✅ Successfully created MultiPromptifyAPI instance!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Step 7: Show package info
    print("\n7. Package information:")
    dist_files = list(Path("dist").glob("*"))
    for file in dist_files:
        size = file.stat().st_size / 1024  # KB
        print(f"   📦 {file.name} ({size:.1f} KB)")
    
    print("\n✅ Build and test completed successfully!")
    print("\nNext steps:")
    print("1. Test with: python examples/api_example.py")
    print("2. Upload to PyPI Test: twine upload --repository testpypi dist/*")
    print("3. Upload to PyPI: twine upload dist/*")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 