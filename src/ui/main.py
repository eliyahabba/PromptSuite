# run_streamlit.py
import argparse
import sys
from pathlib import Path
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MultiPromptify 2.0 - Multi-Prompt Dataset Generator")
    parser.add_argument('--server_port', default=None, type=str, help="Server port")
    parser.add_argument('--step', default=None, type=int, help="Starting step (1-4)")
    parser.add_argument('--debug', type=bool, default=False, help="Enable debug mode")
    args = parser.parse_args()

    # Get the absolute path to load.py
    script_dir = Path(__file__).parent.absolute()
    load_py_path = script_dir / "pages" / "load.py"
    
    # Verify the file exists
    if not load_py_path.exists():
        print(f"Error: load.py not found at {load_py_path}")
        sys.exit(1)
    
    # Set up streamlit arguments for the new interface
    sys.argv = [
        "streamlit", "run", str(load_py_path),
        "--server.address", "localhost",
        "--server.enableCORS", "false", 
        "--server.enableXsrfProtection", "false",
        "--global.suppressDeprecationWarnings", "true",
        "--client.showErrorDetails", "true",
        "--client.toolbarMode", "minimal"
    ]
    
    # Add custom arguments
    if args.server_port:
        sys.argv.extend(["--server.port", args.server_port])
    
    # Add query parameters for step and debug mode
    sys.argv.extend(["--", f"step={args.step or 1}", "--", f"debug={str(args.debug)}"])

    # Launch streamlit
    from streamlit.web import cli as stcli
    sys.exit(stcli.main())