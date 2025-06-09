# run_streamlit.py
import argparse
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', default=None, type=str)
    parser.add_argument('--step', default=None, type=int)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    sys.argv = [
        "streamlit", "run", "load.py",
        "--server.address", "localhost",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--global.suppressDeprecationWarnings", "true",
        "--client.showErrorDetails", "true",
        "--client.toolbarMode", "minimal"
    ]
    sys.argv.extend(["--", f"step={args.step or 1}", "--", f"debug={str(args.debug)}"])

    from streamlit.web import cli as stcli
    sys.exit(stcli.main())