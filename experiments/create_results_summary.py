import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from config import paths
from src.utils.Utils import SaveOutput 

def parse_training_file(file_path):
    """Parse a single training log file and extract final metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Extract final epoch results (last epoch before the summary)
    # Updated pattern: removed "Average cost" line
    epoch_pattern = r'Epoch (\d+) finished! -- Loss: ([\d.]+), Accuracy: ([\d.]+), Val Loss: ([\d.]+), Val Accuracy: ([\d.]+)\nAverage cost -- validation set: ([\d.]+)'
    epochs = re.findall(epoch_pattern, content)
    
    if epochs:
        last_epoch = epochs[-1]
        results['final_epoch'] = int(last_epoch[0])
        results['final_train_loss'] = float(last_epoch[1])
        results['final_train_accuracy'] = float(last_epoch[2])
        results['final_val_loss'] = float(last_epoch[3])
        results['final_val_accuracy'] = float(last_epoch[4])
        results['final_val_average_cost'] = float(last_epoch[5])
    
    # Extract final summary metrics
    # Train metrics - updated patterns
    train_loss_match = re.search(r'Train Loss: ([\d.]+), Train Accuracy: ([\d.]+)', content)
    if train_loss_match:
        results['train_loss'] = float(train_loss_match.group(1))
        results['train_accuracy'] = float(train_loss_match.group(2))
    
    # Updated: Look for "Average cost:" directly after Train Loss line
    train_average_cost_match = re.search(r'Train Loss:.*\nAverage cost: ([\d.]+)', content)
    if train_average_cost_match:
        results['train_average_cost'] = float(train_average_cost_match.group(1))
    
    # Updated: Changed from AECWCE to AEC
    aec_train_match = re.search(r'AEC: ([\d.]+)', content)
    if aec_train_match:
        results['aec_train'] = float(aec_train_match.group(1))

    # RWWCE Train
    rwwce_train_match = re.search(r'RWWCE: ([\d.]+)', content)
    if rwwce_train_match:
        results['rwwce_train'] = float(rwwce_train_match.group(1))
    
    # Validation metrics - updated patterns
    val_loss_match = re.search(r'Validation Loss: ([\d.]+), Validation Accuracy: ([\d.]+)', content)
    if val_loss_match:
        results['val_loss'] = float(val_loss_match.group(1))
        results['val_accuracy'] = float(val_loss_match.group(2))
    
    # Updated: Look for "Average cost:" after Validation Loss line
    val_average_cost_match = re.search(r'Validation Loss:.*\nAverage cost: ([\d.]+)', content)
    if val_average_cost_match:
        results['val_average_cost'] = float(val_average_cost_match.group(1))
    
    # Updated: Changed from AECWCE to AEC - need to get second occurrence
    aec_matches = re.findall(r'AEC: ([\d.]+)', content)
    if len(aec_matches) >= 2:
        results['aec_val'] = float(aec_matches[1])
    
    # RWWCE Validation - need to get second occurrence
    rwwce_matches = re.findall(r'RWWCE: ([\d.]+)', content)
    if len(rwwce_matches) >= 2:
        results['rwwce_val'] = float(rwwce_matches[1])
    
    # Test metrics - updated patterns
    test_loss_match = re.search(r'Test Loss: ([\d.]+), Test Accuracy: ([\d.]+)', content)
    if test_loss_match:
        results['test_loss'] = float(test_loss_match.group(1))
        results['test_accuracy'] = float(test_loss_match.group(2))
    
    # Updated: Look for "Average cost:" after Test Loss line
    test_average_cost_match = re.search(r'Test Loss:.*\nAverage cost: ([\d.]+)', content)
    if test_average_cost_match:
        results['test_average_cost'] = float(test_average_cost_match.group(1))
    
    # Updated: Changed from AECWCE to AEC - need to get third occurrence
    if len(aec_matches) >= 3:
        results['aec_test'] = float(aec_matches[2])
    
    # RWWCE Test - need to get third occurrence
    if len(rwwce_matches) >= 3:
        results['rwwce_test'] = float(rwwce_matches[2])
    
    # Extract False Positives for Class 3 (index 2) from Confusion Matrices
    # False Positives for class 3 = class 0 predicted as 3 + class 1 predicted as 3
    # In confusion matrix: rows are true labels, columns are predictions
    # So we need column 2 (predictions as class 3), excluding row 2 (true class 3)
    
    # Train Confusion Matrix
    train_cm_match = re.search(r'Train Confusion Matrix:\s*\n\s*\[\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\]', content)
    if train_cm_match:
        row0 = [int(x.strip()) for x in train_cm_match.group(1).split()]
        row1 = [int(x.strip()) for x in train_cm_match.group(2).split()]
        row2 = [int(x.strip()) for x in train_cm_match.group(3).split()]
        if len(row0) >= 3 and len(row1) >= 3:
            results['train_fp_class3'] = row0[2] + row1[2]
            print(f"  DEBUG: train_fp_class3 = {results['train_fp_class3']} (row0[2]={row0[2]}, row1[2]={row1[2]})")
        if len(row2) >= 3:
            results['train_tp_class3'] = row2[2]
            print(f"  DEBUG: train_tp_class3 = {results['train_tp_class3']} (row2[2]={row2[2]})")
    
    # Validation Confusion Matrix
    val_cm_match = re.search(r'Validation Confusion Matrix:\s*\n\s*\[\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\]', content)
    if val_cm_match:
        row0 = [int(x.strip()) for x in val_cm_match.group(1).split()]
        row1 = [int(x.strip()) for x in val_cm_match.group(2).split()]
        row2 = [int(x.strip()) for x in val_cm_match.group(3).split()]
        if len(row0) >= 3 and len(row1) >= 3:
            results['val_fp_class3'] = row0[2] + row1[2]
            print(f"  DEBUG: val_fp_class3 = {results['val_fp_class3']} (row0[2]={row0[2]}, row1[2]={row1[2]})")
        if len(row2) >= 3:
            results['val_tp_class3'] = row2[2]
            print(f"  DEBUG: val_tp_class3 = {results['val_tp_class3']} (row2[2]={row2[2]})")
    
    # Test Confusion Matrix
    test_cm_match = re.search(r'Test Confusion Matrix:\s*\n\s*\[\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\]', content)
    if test_cm_match:
        row0 = [int(x.strip()) for x in test_cm_match.group(1).split()]
        row1 = [int(x.strip()) for x in test_cm_match.group(2).split()]
        row2 = [int(x.strip()) for x in test_cm_match.group(3).split()]
        if len(row0) >= 3 and len(row1) >= 3:
            results['test_fp_class3'] = row0[2] + row1[2]
            print(f"  DEBUG: test_fp_class3 = {results['test_fp_class3']} (row0[2]={row0[2]}, row1[2]={row1[2]})")
        if len(row2) >= 3:
            results['test_tp_class3'] = row2[2]
            print(f"  DEBUG: test_tp_class3 = {results['test_tp_class3']} (row2[2]={row2[2]})")

    return results

def analyze_training_results(folder_path):
    """Analyze all training result files in a folder."""
    folder_path = Path(folder_path)
    
    # Get all .txt files in the folder
    txt_files = list(folder_path.glob('*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return None
    
    print(f"Found {len(txt_files)} files to analyze")
    
    # Parse all files
    all_results = []
    for file_path in txt_files:
        try:
            results = parse_training_file(file_path)
            results['filename'] = file_path.name
            all_results.append(results)
            print(f"Processed: {file_path.name}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    if not all_results:
        print("No files were successfully processed")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'final_epoch']  # Exclude epoch number
    
    summary_stats = pd.DataFrame()
    
    for col in numeric_columns:
        if col in df.columns and not df[col].isna().all():
            summary_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
    
    summary_stats = summary_stats.T
    
    return df, summary_stats

def print_summary_report(summary_stats):
    """Print a formatted summary report."""
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    
    # Group metrics by category
    train_metrics = [col for col in summary_stats.index if 'train' in col and 'final' not in col and 'fp_class3' not in col and 'tp_class3' not in col]
    val_metrics = [col for col in summary_stats.index if 'val' in col and 'final' not in col and 'fp_class3' not in col and 'tp_class3' not in col]
    test_metrics = [col for col in summary_stats.index if 'test' in col and 'fp_class3' not in col and 'tp_class3' not in col]
    final_metrics = [col for col in summary_stats.index if 'final' in col and col != 'final_epoch']
    fp_class3_metrics = [col for col in summary_stats.index if 'fp_class3' in col]
    tp_class3_metrics = [col for col in summary_stats.index if 'tp_class3' in col]
    
    categories = [
        ("TRAINING METRICS", train_metrics),
        ("VALIDATION METRICS", val_metrics), 
        ("TEST METRICS", test_metrics),
        ("FINAL EPOCH METRICS", final_metrics),
        ("FALSE POSITIVE COUNT CLASS 3", fp_class3_metrics),
        ("TRUE POSITIVE COUNT CLASS 3", tp_class3_metrics)
    ]
    
    for category_name, metrics in categories:
        if metrics:
            print(f"\n{category_name}:")
            print("-" * len(category_name))
            
            category_df = summary_stats.loc[metrics]
            print(category_df.round(4).to_string())
    
    print("\n" + "="*80)

# Example usage
if __name__ == "__main__":

    # Replace with your folder path (example: random_blobs or steel_plates)
    RESULTS_PATH = paths.BASE_DIR / "artifacts" / "results" / "_random_blobs_balanced" / "Exp_1" / "test_runs_threshold_tuning"
    
    sys.stdout = SaveOutput(RESULTS_PATH / "summary_text.txt")
    # Analyze the results
    df, summary_stats = analyze_training_results(RESULTS_PATH)
    
    if summary_stats is not None:
        # Print the summary report
        print_summary_report(summary_stats)
        
        # Save detailed results to CSV files
        df.to_csv(RESULTS_PATH / 'detailed_results.csv', index=False)
        summary_stats.to_csv(RESULTS_PATH / 'summary_statistics.csv')
        
        print(f"\nDetailed results saved to: detailed_results.csv")
        print(f"Summary statistics saved to: summary_statistics.csv")
        
        # Print some key insights
        print("\nKEY INSIGHTS:")
        print("-" * 20)
        
        # Best performing runs
        if 'test_accuracy' in summary_stats.index:
            best_test_acc_idx = df['test_accuracy'].idxmax()
            print(f"Best test accuracy: {df.loc[best_test_acc_idx, 'test_accuracy']:.4f} (File: {df.loc[best_test_acc_idx, 'filename']})")
        
        if 'test_average_cost' in summary_stats.index:
            best_test_cost_idx = df['test_average_cost'].idxmin()
            print(f"Lowest test cost: {df.loc[best_test_cost_idx, 'test_average_cost']:.4f} (File: {df.loc[best_test_cost_idx, 'filename']})")
        
        # Stability metrics
        if 'test_accuracy' in summary_stats.index:
            cv = summary_stats.loc['test_accuracy', 'std'] / summary_stats.loc['test_accuracy', 'mean']
            print(f"Test accuracy coefficient of variation: {cv:.4f} (lower is more stable)")
    else:
        print("Failed to analyze results. Please check your folder path and file formats.")

    # Restore stdout to normal after execution
    sys.stdout.file.close()
    sys.stdout = sys.stdout.terminal
