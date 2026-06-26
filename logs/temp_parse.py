import re
import os
import glob

def parse_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by both newline and carriage return to get individual tqdm lines
    lines = re.split(r'[\r\n]+', content)
    
    results = {}
    for line in lines:
        match = re.search(r'Iter\s+(\d+)\s+\|\s+Epoch\s+99:.*test_acc=([0-9.]+)', line)
        if match:
            iteration = int(match.group(1))
            test_acc = float(match.group(2))
            results[iteration] = test_acc
            
    return results

if __name__ == '__main__':
    log_dir = '/home/freddyp/projects/def-yani/freddyp/FIRE_playground/logs/DST_cifar10_resnet18_continual_sweep'
    files = glob.glob(os.path.join(log_dir, '*.out'))
    print(f"Total files found: {len(files)}")
    
    incomplete = []
    complete_count = 0
    for f in files:
        res = parse_file(f)
        if len(res) < 10:
            incomplete.append((os.path.basename(f), len(res)))
        else:
            complete_count += 1
            
    print(f"Complete files (10 iterations): {complete_count}")
    print(f"Incomplete files: {len(incomplete)}")
    if incomplete:
        print("First few incomplete files:")
        for inf in incomplete[:10]:
            print(f"  {inf[0]}: {inf[1]} iterations")

