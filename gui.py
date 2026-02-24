#!/usr/bin/env python3
"""
Convenience launcher for the GUI.

Usage:
  python launch_gui.py
  python launch_gui.py examples/result.json
"""
import sys
from hal_analyzer.gui.app import main

if __name__ == "__main__":
    main()
