#!/usr/bin/env python3
"""
Backward compatibility wrapper for the old MultiPromptify main.py
This script shows deprecation warnings and redirects to the new CLI.
"""

import warnings
import sys
import os

def main():
    """Show deprecation warning and redirect to new CLI."""
    warnings.warn(
        "This main.py entry point is deprecated. "
        "MultiPromptify has been redesigned with a new architecture. "
        "Please use the new CLI: 'multipromptify --help' for usage information. "
        "Or import the new API: 'from multipromptify import MultiPromptify'",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("=" * 60)
    print("DEPRECATION NOTICE")
    print("=" * 60)
    print("The old MultiPromptify interface has been replaced.")
    print("Please use the new command-line interface:")
    print()
    print("Basic usage:")
    print("  multipromptify --template \"{instruction:semantic}: {question:paraphrase}\" \\")
    print("                 --data data.csv \\")
    print("                 --instruction \"Answer this question\"")
    print()
    print("For more options:")
    print("  multipromptify --help")
    print()
    print("Python API usage:")
    print("  from multipromptify import MultiPromptify")
    print("  mp = MultiPromptify()")
    print("  variations = mp.generate_variations(template, data, instruction)")
    print()
    print("See README.md for detailed documentation.")
    print("=" * 60)

if __name__ == "__main__":
    main() 