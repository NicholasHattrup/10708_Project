import argparse
import numpy as np


def float_bounds(bounds):

    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} is not a float type literal.")
        
        if x < min(bounds) or x > max(bounds):
            raise argparse.ArgumentTypeError(f"{x} is out of bound {bounds}.")
    
    return restricted_float



def parse_lines(lines, files, fn):
    verified_lines = []
    for i, line in enumerate(lines):
        line = line.strip("\n")
        if line in files:
            verified_lines.append(line)
        else:
            raise Warning(f"Pre-stored filename {line} in not valid in line {i} of file {fn}.")
    return verified_lines


def split_files(split_path, files, args):    
    if args.datasetSplitDone:
        print(f'Gathering dataset split information from {split_path}',flush=True)

        try:
            with open(split_path+"valid_ids.txt", 'r') as f:
                valid_ids = [line.strip("\n") for line in f.readlines()]
            with open(split_path+"test_ids.txt", 'r') as f:
                test_ids = [line.strip("\n") for line in f.readlines()]
            with open(split_path+"train_ids.txt", 'r') as f:
                train_ids = [line.strip("\n") for line in f.readlines()]
            return valid_ids, test_ids, train_ids
        except:
            print(f"No pre-defined split information found. New split info will be used instead.", flush=True)
    
    n_valid, n_test = args.splitRatio.split('_')
    try:
        n_valid = int(n_valid)
    except:
        n_valid = int(float(n_valid) * len(files))
    try:
        n_test = int(n_test)
    except:
        n_test = int(float(n_test) * len(files))
    
    print(f'Creating new split configuration: train({len(files)-n_valid-n_test}), valid ({n_valid}), test({n_test})')
    idx = np.random.permutation(len(files))
    idx = idx.tolist()    
    valid_ids = [files[i] for i in idx[0:n_valid]]
    test_ids = [files[i] for i in idx[n_valid:n_valid+n_test]]
    train_ids = [files[i] for i in idx[n_valid+n_test:]]

    with open(split_path+"valid_ids.txt", "w") as f:
        f.write("\n".join(valid_ids)+"\n")
    with open(split_path+"test_ids.txt", "w") as f:
        f.write("\n".join(test_ids)+"\n")
    with open(split_path+"train_ids.txt", "w") as f:
        f.write("\n".join(train_ids)+"\n")
    print(f"Split configuration is dumped in {split_path}",flush=True)
    
    return valid_ids, test_ids, train_ids
