#!/usr/bin/env python3
"""
Convenience runner — allows using the tool without installing the package.

Usage:
  python halanalyze.py analyze examples/trace_pingpong.json
  python halanalyze.py diff examples/trace_pingpong.json examples/trace_clean.json
"""
import sys
from hal_analyzer.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
