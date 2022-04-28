import argparse
import numpy as np
import os, hashlib
from rdkit.Chem import AllChem as Chem
from sklearn.decomposition import PCA
from joblib import dump, load


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


def GetMD5(ids):
    ids = "".join([id.split('_')[1] for id in ids])
    md5 = hashlib.md5()
    md5.update(ids.encode('utf-8'))
    return md5.hexdigest()


def parse_n_pcs_arg(n_pcs_arg):
    arg = n_pcs_arg.split('_')
    for i, x in enumerate(arg):
        if float(x) >= 1:
            arg[i] = int(x)
        elif float(x) > 0:
            arg[i] = float(x)
        else:
            raise argparse.ArgumentTypeError(f"{n_pcs_arg} cannot be parsed.")
    
    if len(arg) == 2:
        n1, n2 = arg[0], arg[1]
    elif len(arg) == 1:
        n1, n2 = arg[0], arg[0]
    else:
        raise argparse.ArgumentTypeError(f"{n_pcs_arg} cannot be parsed.")
    
    return n1, n2


def smiles_to_fps(smiles, nBits=1024):
    if isinstance(smiles, str):
        smiles = [smiles]
    mols = [Chem.MolFromSmiles(smi, sanitize=True) for smi in smiles]
    mols = [Chem.AddHs(mol) for mol in mols]
    return [Chem.GetMorganFingerprintAsBitVect(mol,2,nBits=nBits) for mol in mols]


def GetCustomizedPCA(libs, n_pcs_arg, verificationKey, modelPath="./data/qm9/", nBits=1024):
    train_lib, valid_lib, test_lib = libs
    node_n, edge_n = parse_n_pcs_arg(n_pcs_arg)
    node_pca, edge_pca = None, None
    for file in os.listdir(modelPath):
        if file.endswith(verificationKey+".joblib") and file.startswith("node"):
            node_pca = load(modelPath+file)
            print(f"Loading pre-trained PCA model for nodes...", flush=True)
        if file.endswith(verificationKey+".joblib") and file.startswith("edge"):
            edge_pca = load(modelPath+file)
            print(f"Loading pre-trained PCA model for edges...", flush=True)
    
    train_lib.init_reduction()
    train_fragments = train_lib.fragment_library
    train_linkages = train_lib.linkage_library

    if node_pca is None:
        print(f"PCA model for nodes not found. Start training and saving...")
        train_frag_fps = smiles_to_fps(train_fragments, nBits)
        node_pca = PCA(n_components=64).fit(train_frag_fps)
        dump(node_pca, modelPath+"node_"+verificationKey+".joblib")
    if edge_pca is None:
        print(f"PCA model for edges not found. Start training and saving...")
        train_link_fps = smiles_to_fps(train_linkages, nBits)
        edge_pca = PCA(n_components=64).fit(train_link_fps)
        dump(node_pca, modelPath+"edge_"+verificationKey+".joblib")
    
    node_cev = [sum(node_pca.explained_variance_ratio_[:i]) for i in range(64)] # cummulative explained variance
    edge_cev = [sum(edge_pca.explained_variance_ratio_[:i]) for i in range(64)]
    if isinstance(node_n, float):
        for i, x in enumerate(node_cev):
            if x < node_n:
                node_n = i+1
                break
    if isinstance(edge_n, float):
        for i, x in enumerate(edge_cev):
            if x < edge_n:
                edge_n = i+1
                break
    print(f"Number of PCs: node ({node_n}, {node_cev[node_n-1]:.4f}) \t edge ({edge_n}, {edge_cev[edge_n-1]:.4f})", flush=True)

    fragments, linkages = train_fragments, train_linkages
    valid_lib.init_reduction()
    fragments = fragments.union(valid_lib.fragment_library)
    linkages = linkages.union(valid_lib.linkage_library)
    test_lib.init_reduction()
    fragments = list(fragments.union(test_lib.fragment_library))
    linkages = list(linkages.union(test_lib.linkage_library))

    frag_fps = node_pca.transform(smiles_to_fps(fragments))[:, :node_n]
    link_fps = node_pca.transform(smiles_to_fps(linkages))[:, :edge_n]

    frag_fp_dict = {fragments[i]: frag_fps[i] for i in range(len(fragments))}
    link_fp_dict = {linkages[i]: link_fps[i] for i in range(len(linkages))}

    def NodeConverter(x): return frag_fp_dict[x]
    def EdgeConverter(x): return link_fp_dict[x]

    return NodeConverter, EdgeConverter

      

