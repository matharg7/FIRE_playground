#!/usr/bin/env python3
import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_filename(filename):
    """
    Parses parameters from a filename like:
    RESNET18_CIFAR10_rigl_s0.1_pr0.1_nmu10.out
    Returns a dict of parameters or None if parsing fails.
    """
    basename = os.path.basename(filename)
    pattern = r'RESNET18_CIFAR10_(?P<sparsifier>[a-zA-Z0-9]+)_s(?P<sparsity>[0-9.]+)_pr(?P<pruning_ratio>[0-9.]+)_nmu(?P<nmu>\d+)\.out'
    match = re.match(pattern, basename)
    if match:
        return {
            'sparsifier': match.group('sparsifier'),
            'sparsity': float(match.group('sparsity')),
            'pruning_ratio': float(match.group('pruning_ratio')),
            'num_mask_updates': int(match.group('nmu')),
            'filepath': filename,
            'basename': basename
        }
    return None

def parse_log_file(filepath):
    """
    Extracts test accuracies per epoch from the log file.
    Returns a list of (step, test_acc) where step = iteration * 100 + epoch.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by lines (handling carriage returns from tqdm updates)
    lines = re.split(r'[\r\n]+', content)
    
    # Map (iteration, epoch) to test_acc, keeping the last seen (most complete) value
    results = {}
    for line in lines:
        match = re.search(r'Iter\s+(\d+)\s+\|\s+Epoch\s+(\d+):.*test_acc=([0-9.]+)', line)
        if match:
            iteration = int(match.group(1))
            epoch = int(match.group(2))
            test_acc = float(match.group(3))
            results[(iteration, epoch)] = test_acc
            
    # Convert to sorted list of steps
    sorted_steps = []
    for (iteration, epoch), test_acc in sorted(results.items()):
        step = iteration * 100 + epoch
        sorted_steps.append((step, test_acc))
        
    return sorted_steps

def main():
    parser = argparse.ArgumentParser(
        description="Plot Test Accuracy vs Steps for Continual Learning Sparse Training Sweep."
    )
    parser.add_argument(
        "--log-dir", 
        default="/home/freddyp/projects/def-yani/freddyp/FIRE_playground/logs/DST_cifar10_resnet18_continual_sweep",
        help="Path to directory containing .out log files"
    )
    parser.add_argument(
        "--sparsifier", 
        default="rigl",
        help="Sparsifier algorithm (e.g. rigl, set). Can be comma-separated, 'all', or '*' (default: rigl)."
    )
    parser.add_argument(
        "--pruning-ratio", 
        default="0.3",
        help="Pruning ratio (e.g. 0.3). Can be comma-separated, 'all', or '*' (default: 0.3)."
    )
    parser.add_argument(
        "--num-mask-updates", 
        default="1000",
        help="Number of mask updates (e.g. 1000). Can be comma-separated, 'all', or '*' (default: 1000)."
    )
    parser.add_argument(
        "--sparsity", 
        default="0.9",
        help="Sparsity target (e.g. 0.9). Can be comma-separated, 'all', or '*' (default: 0.9)."
    )
    parser.add_argument(
        "--x-interval", 
        type=float, 
        help="Tick interval/spacing for the X-axis (steps)"
    )
    parser.add_argument(
        "--y-interval", 
        type=float, 
        help="Tick interval/spacing for the Y-axis (accuracy)"
    )
    parser.add_argument(
        "--x-lim", 
        nargs=2, 
        type=float, 
        help="X-axis limit (min max)"
    )
    parser.add_argument(
        "--y-lim", 
        nargs=2, 
        type=float, 
        help="Y-axis limit (min max)"
    )
    parser.add_argument(
        "--output", 
        default="accuracy_plot.png",
        help="Filename to save the generated plot"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Display the plot window using plt.show()"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive prompt mode to select filter values."
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.log_dir):
        print(f"Error: Log directory not found at {args.log_dir}")
        return
        
    # Scan all log files
    all_files = glob.glob(os.path.join(args.log_dir, "*.out"))
    parsed_runs = []
    for f in all_files:
        parsed = parse_filename(f)
        if parsed:
            parsed_runs.append(parsed)
            
    if not parsed_runs:
        print(f"No matching log files found in {args.log_dir}")
        return
        
    print(f"Scanned {len(parsed_runs)} valid log files.")
    
    # Helper to parse list inputs
    def parse_filter_value(val, value_type):
        if val is None:
            return None
        if not isinstance(val, str):
            return [value_type(val)]
        if val.strip().lower() in ('all', '*'):
            return None
        return [value_type(x.strip()) for x in val.split(',')]

    # Parse filters / Prompt in interactive mode
    if args.interactive:
        print("\n--- Interactive Filter Mode ---")
        unique_sparsifiers = sorted(list(set(r['sparsifier'] for r in parsed_runs)))
        unique_sparsities = sorted(list(set(r['sparsity'] for r in parsed_runs)))
        unique_prs = sorted(list(set(r['pruning_ratio'] for r in parsed_runs)))
        unique_nmus = sorted(list(set(r['num_mask_updates'] for r in parsed_runs)))
        
        print(f"Available sparsifiers: {unique_sparsifiers}")
        val = input(f"Select sparsifier(s) (comma-separated or Enter for all, default: rigl): ")
        if not val.strip():
            filter_sparsifiers = [ "rigl" ]
        else:
            filter_sparsifiers = parse_filter_value(val, str)
        
        print(f"Available sparsities: {unique_sparsities}")
        val = input(f"Select sparsity/sparsities (comma-separated or Enter for all, default: 0.9): ")
        if not val.strip():
            filter_sparsity = [ 0.9 ]
        else:
            filter_sparsity = parse_filter_value(val, float)
        
        print(f"Available pruning ratios: {unique_prs}")
        val = input(f"Select pruning ratio(s) (comma-separated or Enter for all, default: 0.3): ")
        if not val.strip():
            filter_pr = [ 0.3 ]
        else:
            filter_pr = parse_filter_value(val, float)
        
        print(f"Available num-mask-updates (nmu): {unique_nmus}")
        val = input(f"Select num-mask-updates (comma-separated or Enter for all, default: 1000): ")
        if not val.strip():
            filter_nmu = [ 1000 ]
        else:
            filter_nmu = parse_filter_value(val, int)
    else:
        # Parse filters from CLI arguments
        filter_sparsifiers = parse_filter_value(args.sparsifier, str)
        filter_pr = parse_filter_value(args.pruning_ratio, float)
        filter_nmu = parse_filter_value(args.num_mask_updates, int)
        filter_sparsity = parse_filter_value(args.sparsity, float)
        
    # Apply filtering
    matched_runs = []
    for r in parsed_runs:
        if filter_sparsifiers and r['sparsifier'] not in filter_sparsifiers:
            continue
        if filter_sparsity and r['sparsity'] not in filter_sparsity:
            continue
        if filter_pr and r['pruning_ratio'] not in filter_pr:
            continue
        if filter_nmu and r['num_mask_updates'] not in filter_nmu:
            continue
        matched_runs.append(r)
        
    if not matched_runs:
        print("No log files match the specified filters.")
        return
        
    print(f"\nPlotting {len(matched_runs)} run(s):")
    
    plt.figure(figsize=(10, 6))
    
    for r in sorted(matched_runs, key=lambda x: (x['sparsifier'], x['sparsity'], x['pruning_ratio'], x['num_mask_updates'])):
        print(f"  - {r['basename']}")
        data = parse_log_file(r['filepath'])
        if not data:
            print(f"    WARNING: No test accuracy data parsed from {r['basename']}")
            continue
            
        steps, accs = zip(*data)
        label = f"{r['sparsifier']} (s={r['sparsity']}, pr={r['pruning_ratio']}, nmu={r['num_mask_updates']})"
        plt.plot(steps, accs, label=label, alpha=0.8)
        
    plt.title("Test Accuracy vs Steps (Global Epoch)", y=1.15)
    plt.xlabel("Steps (Global Epoch)")
    plt.ylabel("Test Accuracy")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Place legend below the title in the middle (above the plot frame)
    ncol_val = min(3, len(matched_runs)) if len(matched_runs) > 0 else 1
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=ncol_val, frameon=True)
    
    # Adjust ticks/spacing
    ax = plt.gca()
    if args.x_interval is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(args.x_interval))
    if args.y_interval is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(args.y_interval))
        
    if args.x_lim:
        plt.xlim(args.x_lim)
    if args.y_lim:
        plt.ylim(args.y_lim)
        
    plt.tight_layout()
    
    # Save output
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot successfully saved to: {os.path.abspath(args.output)}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
