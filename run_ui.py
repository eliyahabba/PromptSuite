#!/usr/bin/env python3
"""
Simple script to run MultiPromptify UI
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the MultiPromptify UI"""
    print("🚀 MultiPromptify - Multi-Prompt Dataset Generator")
    print("=" * 50)
    
    # Check if we're in the right directory
    ui_main = Path("src/ui/main.py")
    if not ui_main.exists():
        print("❌ Error: Please run this script from the project root directory")
        print(f"Looking for: {ui_main.absolute()}")
        sys.exit(1)
    
    print("🌟 Starting MultiPromptify UI...")
    print("📱 The interface will open in your browser")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Add src to Python path and run the UI
        subprocess.run([
            sys.executable, 
            str(ui_main),
            "--server_port", "8501"
        ], env={**os.environ, "PYTHONPATH": str(Path("src").absolute())})
    except KeyboardInterrupt:
        print("\n👋 UI stopped. Thanks for using MultiPromptify!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 