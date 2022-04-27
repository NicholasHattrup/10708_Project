import types
from sklearn.decomposition import PCA
#from matplotlib.pyplot import get
import numpy as np
#import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import GetSymmSSSR as SSSR
import networkx as nx


PTABLE = Chem.GetPeriodicTable()

def get_atom(mol, idx):
    atom = mol.GetAtomWithIdx(idx)
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())

    if atom.GetIsAromatic() and atom.GetSymbol() == 'N':
        new_atom.SetNumExplicitHs(atom.GetTotalNumHs())
    
    return new_atom

def get_substruct_for_edge(mol, atoms):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    new_atom_map = {}
    for idx in atoms:
        new_atom = get_atom(mol, idx)
        new_atom_map[idx] = new_atom
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if (a1 in atoms) and (a2 in atoms):
            new_mol.AddBond(new_atom_map[a1], new_atom_map[a2], bond.GetBondType())
    try:
        new_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(new_smi)
        new_mol = Chem.Sanitize(new_mol.GetMol())
        return new_smi, new_mol
    except:
        raise Warning(f"New molecule creation failed.")
        return None, None


X = np.random.randint((5,5))