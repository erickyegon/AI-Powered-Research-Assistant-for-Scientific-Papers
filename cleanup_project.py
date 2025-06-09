#!/usr/bin/env python3
"""
Project Cleanup Script for AI Research Assistant
This script removes unnecessary files and reorganizes the project structure.
"""

import os
import shutil
import sys
from pathlib import Path

def confirm_action(message):
    """Ask for user confirmation."""
    response = input(f"{message} (y/N): ").lower().strip()
    return response == 'y'

def safe_remove(path):
    """Safely remove a file or directory."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"✅ Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✅ Removed directory: {path}")
        else:
            print(f"⚠️ Path not found: {path}")
    except Exception as e:
        print(f"❌ Error removing {path}: {e}")

def safe_rename(old_path, new_path):
    """Safely rename a file."""
    try:
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {old_path} -> {new_path}")
        else:
            print(f"⚠️ File not found: {old_path}")
    except Exception as e:
        print(f"❌ Error renaming {old_path}: {e}")

def main():
    """Main cleanup function."""
    print("🧹 AI Research Assistant - Project Cleanup")
    print("=" * 50)
    
    # Get current directory
    project_root = Path.cwd()
    print(f"📁 Project root: {project_root}")
    
    if not confirm_action("⚠️ This will delete files permanently. Continue?"):
        print("❌ Cleanup cancelled.")
        return
    
    print("\n🗑️ Starting cleanup...")
    
    # Files to delete (root level)
    root_files_to_delete = [
        "minimal_test.py",
        "project_template.py", 
        "test_app.py",
        "test_requirements.txt",
        "start_professional.py",
        "deploy.sh",
        "start.sh"
    ]
    
    print("\n📁 Cleaning root directory...")
    for file in root_files_to_delete:
        safe_remove(file)
    
    # Backend files to delete
    backend_files_to_delete = [
        "backend/main.py",  # Keep main_fresh.py as the main
        "backend/main_professional.py",
        "backend/main_simple.py", 
        "backend/main_working.py",
        "backend/simple_main.py",
        "backend/langserve_app.py",
        "backend/validate_env.py",
        "backend/render_build.sh"
    ]
    
    print("\n📁 Cleaning backend directory...")
    for file in backend_files_to_delete:
        safe_remove(file)
    
    # Backend directories to delete (KEEP LangChain components)
    backend_dirs_to_delete = [
        "backend/__pycache__",
        # NOTE: Keeping backend/chains and backend/graphs for LangChain/LangGraph
    ]

    for directory in backend_dirs_to_delete:
        if os.path.exists(directory):
            safe_remove(directory)
    
    # Utils files to potentially delete
    utils_files_to_check = [
        "backend/utils/__pycache__",
        "backend/utils/query_processor.py",  # Duplicate
        "backend/utils/db.py",  # If not using database
        "backend/utils/auth.py",  # If not using auth
        "backend/utils/memory.py",  # If not using caching
        "backend/utils/embedding.py"  # Duplicate functionality
    ]
    
    print("\n📁 Cleaning utils directory...")
    for file in utils_files_to_check:
        if os.path.exists(file):
            if confirm_action(f"Delete {file}?"):
                safe_remove(file)
    
    # Tools cache
    safe_remove("backend/tools/__pycache__")
    
    # Frontend files to delete
    frontend_files_to_delete = [
        "frontend/app_simple.py"  # Keep main app.py
    ]
    
    print("\n📁 Cleaning frontend directory...")
    for file in frontend_files_to_delete:
        if confirm_action(f"Delete {file}?"):
            safe_remove(file)
    
    # Frontend directories to potentially delete
    frontend_dirs_to_check = [
        "frontend/components",
        "frontend/assets"
    ]
    
    for directory in frontend_dirs_to_check:
        if os.path.exists(directory):
            if confirm_action(f"Delete {directory}? (unused components)"):
                safe_remove(directory)
    
    # Virtual environment (should never be committed)
    if os.path.exists("research_env"):
        if confirm_action("Delete research_env/ virtual environment?"):
            safe_remove("research_env")
    
    # Rename main_fresh.py to main.py
    print("\n🔄 Renaming files...")
    if os.path.exists("backend/main_fresh.py"):
        if confirm_action("Rename backend/main_fresh.py to backend/main.py?"):
            safe_rename("backend/main_fresh.py", "backend/main.py")
    
    print("\n✅ Cleanup complete!")
    print("\n📋 Recommended final structure:")
    print("""
    AI-Powered Research Assistant/
    ├── README.md
    ├── requirements.txt
    ├── docker-compose.yml
    ├── .env (create this)
    ├── backend/
    │   ├── main.py (renamed from main_fresh.py)
    │   ├── requirements.txt
    │   ├── Dockerfile
    │   ├── __init__.py
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   └── eur_client.py
    │   └── tools/
    │       ├── __init__.py
    │       ├── embedding_tool.py
    │       └── query_processor.py
    └── frontend/
        ├── app.py
        ├── requirements.txt
        ├── Dockerfile
        └── __init__.py
    """)
    
    print("\n🚀 Next steps:")
    print("1. Test the application: uvicorn backend.main:app --reload")
    print("2. Test the frontend: streamlit run frontend/app.py")
    print("3. Create .env file with your API keys")
    print("4. Update README.md with new structure")

if __name__ == "__main__":
    main()
