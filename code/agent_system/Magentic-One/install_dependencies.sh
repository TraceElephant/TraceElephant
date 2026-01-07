#!/bin/bash

echo "Install dependencies for GAIA Benchmark..."
echo "================================"

# Install basic dependencies
echo "Install python-dotenv..."
pip install python-dotenv

echo ""
echo "Install autogen related packages..."
pip install pyautogen autogen-agentchat autogen-ext

echo ""
echo "================================"
echo "Dependencies installed!"
echo ""
echo "Run test to verify:"
echo "  python test_gaia_setup.py"
