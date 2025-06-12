#!/usr/bin/env python3
"""
Demo script for MultiPromptify 2.0 UI
This script demonstrates the new streamlined interface.
"""

import subprocess
import sys
import os

def main():
    """Run the MultiPromptify 2.0 UI demo"""
    print("🚀 MultiPromptify 2.0 UI Demo")
    print("=" * 50)
    print()
    print("This new UI provides a streamlined 4-step process:")
    print("1. 📁 Upload Data - Load CSV/JSON or use sample datasets")
    print("2. 🔧 Template Builder - Create templates with variation annotations")
    print("3. ⚡ Generate Variations - Configure and generate prompt variations")
    print("4. 🎉 View Results - Analyze, search, and export your variations")
    print()
    print("Key Features:")
    print("✅ Template suggestions based on your data")
    print("✅ Real-time template validation and preview")
    print("✅ Sample datasets for quick testing")
    print("✅ Advanced filtering and search")
    print("✅ Multiple export formats (JSON, CSV, TXT, Custom)")
    print("✅ No manual annotation required!")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/ui/run_streamlit.py"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    choice = input("Would you like to launch the UI? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        print("\n🚀 Launching MultiPromptify 2.0 UI...")
        print("The interface will open in your browser.")
        print("Press Ctrl+C to stop the server when done.")
        print()
        
        try:
            # Launch the UI
            subprocess.run([
                sys.executable, 
                "src/ui/run_streamlit.py", 
                "--step=1"
            ])
        except KeyboardInterrupt:
            print("\n👋 UI stopped. Thanks for trying MultiPromptify 2.0!")
        except Exception as e:
            print(f"❌ Error launching UI: {e}")
            print("Please make sure Streamlit is installed: pip install streamlit")
    else:
        print("👋 To run the UI later, use: python src/ui/run_streamlit.py")

if __name__ == "__main__":
    main() 