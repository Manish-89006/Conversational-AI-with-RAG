#!/usr/bin/env python3
"""
Setup script for the Conversational AI with RAG system.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "chroma_db",
        "logs",
        "uploads",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create environment file from template."""
    print("\n🔧 Setting up environment configuration...")
    
    env_template = "env_example.txt"
    env_file = ".env"
    
    if os.path.exists(env_file):
        print(f"⚠️  {env_file} already exists, skipping creation")
        return True
    
    if os.path.exists(env_template):
        shutil.copy(env_template, env_file)
        print(f"✅ Created {env_file} from template")
        print("⚠️  Please edit .env file with your API keys")
        return True
    else:
        print(f"❌ Template file {env_template} not found")
        return False

def validate_setup():
    """Validate the setup."""
    print("\n🔍 Validating setup...")
    
    # Check if required files exist
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/llm_manager.py",
        "src/vector_store.py",
        "src/rag_pipeline.py",
        "src/document_processor.py",
        "src/api.py",
        "main.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    # Try to import main modules
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.config import config
        print("✅ Configuration module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Conversational AI with RAG - Setup\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create environment file
    create_env_file()
    
    # Validate setup
    if not validate_setup():
        print("\n❌ Setup validation failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run the demo: python demo.py")
    print("3. Start the web interface: python main.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()

