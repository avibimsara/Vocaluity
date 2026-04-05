#!/bin/bash

echo "=============================================="
echo "  VOCALUITY - AI Vocal Detection Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Check Python
echo "Step 1: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
print_status "Dependencies installed"

# Step 4: Create directory structure
echo ""
echo "Step 4: Creating directory structure..."
mkdir -p data/raw/fakemusiccaps
mkdir -p data/raw/real
mkdir -p data/raw/ai_generated
mkdir -p data/processed
mkdir -p data/features
mkdir -p models
mkdir -p results
mkdir -p notebooks
print_status "Directories created"

# Step 5: Install ffmpeg (for audio processing)
echo ""
echo "Step 5: Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    print_status "ffmpeg is installed"
else
    print_warning "ffmpeg not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        print_error "Please install ffmpeg manually"
    fi
fi

# Step 6: Create __init__.py files
touch src/__init__.py
touch app/__init__.py

# Step 7: Print next steps
echo ""
echo "=============================================="
echo "  SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download the FakeMusicCaps dataset:"
echo "   → Go to: https://zenodo.org/records/15063698"
echo "   → Extract to: data/raw/fakemusiccaps/"
echo ""
echo "2. Or use your own data:"
echo "   → Put real audio in: data/raw/real/"
echo "   → Put AI audio in: data/raw/ai_generated/"
echo ""
echo "3. Train the model:"
echo "   → python src/train.py"
echo ""
echo "4. Launch the web interface:"
echo "   → streamlit run app/streamlit_app.py"
echo ""
echo "=============================================="
