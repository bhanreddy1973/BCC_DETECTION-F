import os
import shutil
from pathlib import Path

def remove_empty_logs():
    """Remove empty log files"""
    log_dirs = ['logs', 'models/checkpoints/logs']
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.endswith('.log'):
                    file_path = os.path.join(log_dir, file)
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)
                        print(f"Removed empty log file: {file_path}")

def remove_duplicate_files():
    """Remove duplicate or unnecessary files"""
    files_to_remove = [
        'models/checkpoints/final_model.h5',  # best_model.h5 is sufficient
        'README.pdf',  # README.md is sufficient
        'bcc-detection-prompt.md',
        'bcc-detection-prompt.pdf',
        'optimized_pipeline.log',
        'feature_extraction.log'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed file: {file}")

def clean_empty_directories():
    """Remove empty directories"""
    directories = [
        'visualizations',
        'docs/images',
        'models/checkpoints/fold_1',
        'models/checkpoints/logs'
    ]
    
    for directory in directories:
        if os.path.exists(directory) and not os.listdir(directory):
            os.rmdir(directory)
            print(f"Removed empty directory: {directory}")

def main():
    """Main cleanup function"""
    print("Starting cleanup...")
    
    # Remove empty log files
    remove_empty_logs()
    
    # Remove duplicate and unnecessary files
    remove_duplicate_files()
    
    # Clean empty directories
    clean_empty_directories()
    
    print("\nCleanup completed!")

if __name__ == '__main__':
    main() 