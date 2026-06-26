#!/usr/bin/env python3
import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DENSE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "Dense", "RESNET18_CIFAR10_dense.out")


global def_sparsifier
global def_sparsity 
global def_nmu 
global def_pr
global def_variable
global output 

def set_default(sp=None, s=None, nmu=None, pr=None, variable="pruning_ratio"):
    global def_sparsifier
    global def_sparsity 
    global def_nmu 
    global def_pr
    global def_variable
    global output 
    if sp is not None:
        def_sparsifier = sp
    else:
        def_sparsifier = " "
    if s is not None:
        def_sparsity = s
    else:
        def_sparsity = " "
    if nmu is not None:
        def_nmu = nmu
    else:
        def_nmu = " "
    if pr is not None:           
        def_pr = pr
    else:
        def_pr = " "
    if variable is not None:
        def_variable = variable
    output = "plot-dense/"+def_sparsifier + "_" + str(def_sparsity) + "_" + str(def_nmu) + "_" + str(def_pr) + "_var" + variable + ".png"

    

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


# Mapping from variable name to the key used in parsed run dicts and display labels
VARIABLE_INFO = {
    'sparsity':         {'key': 'sparsity',          'label': 'Sparsity',          'short': 's',   'type': float},
    'pruning_ratio':    {'key': 'pruning_ratio',     'label': 'Pruning Ratio',     'short': 'pr',  'type': float},
    'num_mask_updates': {'key': 'num_mask_updates',  'label': 'Num Mask Updates',  'short': 'nmu', 'type': int},
}


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
        default=def_sparsifier,
        help="Sparsifier algorithm (e.g. rigl, set)."
    )
    parser.add_argument(
        "--sparsity", 
        default=def_sparsity,
        help="Fixed sparsity target (e.g. 0.9). Ignored when --variable=sparsity."
    )
    parser.add_argument(
        "--pruning-ratio", 
        default=def_pr,
        help="Fixed pruning ratio (e.g. 0.3). Ignored when --variable=pruning_ratio."
    )
    parser.add_argument(
        "--num-mask-updates", 
        default=def_nmu,
        help="Fixed number of mask updates (e.g. 1000). Ignored when --variable=num_mask_updates."
    )
    parser.add_argument(
        "--variable",
        default=def_variable,
        choices=['sparsity', 'pruning_ratio', 'num_mask_updates'],
        help="Which hyperparameter to vary (produces multiple lines). The other two are held fixed."
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
        default=output,
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
    
    # Determine the variable hyperparameter
    variable = args.variable
    var_info = VARIABLE_INFO[variable]
    
    # The two fixed hyperparameters (excluding the variable one)
    fixed_params = {k: v for k, v in VARIABLE_INFO.items() if k != variable}
    
    # Parse fixed filter values
    if args.interactive:
        print("\n--- Interactive Filter Mode ---")
        unique_sparsifiers = sorted(set(r['sparsifier'] for r in parsed_runs))
        print(f"Available sparsifiers: {unique_sparsifiers}")
        val = input(f"Select sparsifier (default: rigl): ").strip()
        filter_sparsifier = val if val else "rigl"
        
        print(f"\nVariable hyperparameter: {var_info['label']} (all values will be plotted)")
        
        for param_name, pinfo in fixed_params.items():
            unique_vals = sorted(set(r[pinfo['key']] for r in parsed_runs))
            print(f"\nAvailable {pinfo['label']} values: {unique_vals}")
            val = input(f"Select {pinfo['label']} (single value): ").strip()
            if val:
                pinfo['_filter'] = pinfo['type'](val)
            else:
                # Use the CLI default
                if param_name == 'sparsity':
                    pinfo['_filter'] = pinfo['type'](args.sparsity)
                elif param_name == 'pruning_ratio':
                    pinfo['_filter'] = pinfo['type'](args.pruning_ratio)
                elif param_name == 'num_mask_updates':
                    pinfo['_filter'] = pinfo['type'](args.num_mask_updates)
    else:
        filter_sparsifier = args.sparsifier
        for param_name, pinfo in fixed_params.items():
            if param_name == 'sparsity':
                pinfo['_filter'] = pinfo['type'](args.sparsity)
            elif param_name == 'pruning_ratio':
                pinfo['_filter'] = pinfo['type'](args.pruning_ratio)
            elif param_name == 'num_mask_updates':
                pinfo['_filter'] = pinfo['type'](args.num_mask_updates)
    
    # Apply filtering: match sparsifier + fixed params, allow variable to vary
    matched_runs = []
    for r in parsed_runs:
        if r['sparsifier'] != filter_sparsifier:
            continue
        skip = False
        for param_name, pinfo in fixed_params.items():
            if r[pinfo['key']] != pinfo['_filter']:
                skip = True
                break
        if skip:
            continue
        matched_runs.append(r)
        
    if not matched_runs:
        print("No log files match the specified filters.")
        return
    
    # Sort by the variable hyperparameter value
    matched_runs.sort(key=lambda x: x[var_info['key']])
    
    # Build subtitle showing fixed params
    fixed_parts = [f"sparsifier={filter_sparsifier}"]
    for param_name, pinfo in fixed_params.items():
        fixed_parts.append(f"{pinfo['short']}={pinfo['_filter']}")
    subtitle = ", ".join(fixed_parts)
    
    print(f"\nPlotting {len(matched_runs)} run(s), varying {var_info['label']}:")
        
    plt.figure(figsize=(10, 6))
    
    # Plot dense baseline
    if os.path.isfile(DENSE_LOG_PATH):
        dense_data = parse_log_file(DENSE_LOG_PATH)
        if dense_data:
            dense_steps, dense_accs = zip(*dense_data)
            plt.plot(dense_steps, dense_accs, label="Dense", color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=10)
    
    for r in matched_runs:
        print(f"  - {r['basename']}")
        data = parse_log_file(r['filepath'])
        if not data:
            print(f"    WARNING: No test accuracy data parsed from {r['basename']}")
            continue
            
        steps, accs = zip(*data)
        var_val = r[var_info['key']]
        label = f"{var_info['short']}={var_val}"
        plt.plot(steps, accs, label=label, alpha=0.8)
        
    plt.title(f"Test Accuracy vs Steps — varying {var_info['label']}\n({subtitle})", y=1.15)
    plt.xlabel("Steps (Global Epoch)")
    plt.ylabel("Test Accuracy")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Place legend below the title in the middle (above the plot frame)
    total_lines = len(matched_runs) + (1 if os.path.isfile(DENSE_LOG_PATH) else 0)
    ncol_val = min(5, total_lines) if total_lines > 0 else 1
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
    set_default("rigl", 0.9, 1000, 0.3, "pruning_ratio")
    main()
