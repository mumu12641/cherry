#!/bin/bash

BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BUILD_DIR="$SCRIPT_DIR/../build"
echo -e "${BLUE} 🔨 Starting Compile core.so${NC}"
cd "$BUILD_DIR"
ninja cherry_py

echo -e "${YELLOW} 🏃 Running py...${NC}"
cd "$SCRIPT_DIR"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python test.py
