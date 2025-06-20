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
    
    # Check if we're in the right directory (go up one level from scripts/)
    project_root = Path(__file__).parent.parent
    ui_main = project_root / "src/multipromptify/ui/main.py"
    if not ui_main.exists():
        print("❌ Error: Could not find MultiPromptify UI files")
        print(f"Looking for: {ui_main.absolute()}")
        sys.exit(1)
    
    print("🌟 Starting MultiPromptify UI...")
    print("📱 The interface will open in your browser")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Add src to Python path and run the UI
        src_path = project_root / "src"
        subprocess.run([
            sys.executable, 
            str(ui_main),
            "--server_port", "8501"
        ], env={**os.environ, "PYTHONPATH": str(src_path.absolute())})
    except KeyboardInterrupt:
        print("\n👋 UI stopped. Thanks for using MultiPromptify!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 