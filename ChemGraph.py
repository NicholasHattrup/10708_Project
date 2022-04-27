import types
from sklearn.decomposition import PCA
#from matplotlib.pyplot import get
import numpy as np
#import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import GetSymmSSSR as SSSR
import networkx as nx


PTABLE = Chem.GetPeriodicTable()

def bond_to_str(bond_type):
    bt = str(bond_type).lower()
    if bt=='single':
        return ''
    elif bt=='double':
        return '='
    elif bt=='triple':
        return '#'
    else:
        raise Warning(f"Uexpected aromatic ring found. Check again.")
        return None

def sort_ring(mol, atoms):
    bond_type = mol.GetBondBetweenAtoms(atoms[0], atoms[-1]).GetBondType()
    if str(bond_type).lower()=='single':
        return atoms
    else:
        return sort_ring(mol, atoms[1:] + [atoms[0]])
    
def get_substruct(smi, atoms):
    mol = Chem.MolFromSmiles(smi, sanitize=False)

    if len(atoms) == 1:
        new_smi = mol.GetAtomWithIdx(atoms[0]).GetSymbol()
    elif len(atoms) == 2:
        symbol_1 = mol.GetAtomWithIdx(atoms[0]).GetSymbol()
        symbol_2 = mol.GetAtomWithIdx(atoms[1]).GetSymbol()
        bond_type = mol.GetBondBetweenAtoms(atoms[0], atoms[1]).GetBondType()
        new_smi = symbol_1 + bond_to_str(bond_type) + symbol_2
    elif len(atoms) > 2:
        
        bt = str(mol.GetBondBetweenAtoms(atoms[0], atoms[1]).GetBondType()).lower()
        if bt=='aromatic':
            try:
                symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atoms]
                new_smi = "".join(symbols).lower()
                new_smi = new_smi[0] + '1' + new_smi[1:] + '1'
                new_mol = Chem.MolFromSmiles(new_smi)
                Chem.AddHs(new_mol)
                new_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
                return new_smi, new_mol
            except:
                pass
        atoms = sort_ring(mol, atoms)
        new_smi = mol.GetAtomWithIdx(atoms[0]).GetSymbol()
        for i in range(len(atoms)-1):
            bond_type = mol.GetBondBetweenAtoms(atoms[i], atoms[i+1]).GetBondType()
            new_smi += bond_to_str(bond_type) + mol.GetAtomWithIdx(atoms[i+1]).GetSymbol()
        new_smi = new_smi[0] + '1' + new_smi[1:] + '1'
    
    new_mol = Chem.MolFromSmiles(new_smi, sanitize=True)
    Chem.AddHs(new_mol)
    new_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
    return new_smi, new_mol

class SubstructGraph(object):
    """
    Substructure graph of a molecule. Hydrogen atoms are ignored when building the graph but may be considered in features.
    """
    def __init__(self, xyz_file, Hs_in_fragments=True, Hs_in_linkages=True):
        self.filepath = xyz_file
        self.filaname = xyz_file.split("/")[-1]
        self.parse_xyz_file()  # self.n_atoms, self.smi, self.atoms, self.coord, self.gap
        self.mol = Chem.MolFromSmiles(self.smi)
        self.mol_Hs = Chem.AddHs(self.mol)

        self.fragments = []
        self.Hs_in_fragments = Hs_in_fragments
        self.linkages = []
        self.Hs_in_linkages = Hs_in_linkages
        self.graph = self.init_graph(Hs_in_fragments, Hs_in_linkages)   # nx graph object
    
    def parse_xyz_file(self):
        self.atoms, self.atomic_nums, self.coords = [], [], []
        with open(self.filepath, "r") as f:
            self.n_atoms = int(f.readline())
            self.gap = float(f.readline().split()[-1])
            for _ in range(self.n_atoms):
                line = f.readline().split()
                self.atoms.append(line[0])
                self.atomic_nums.append(PTABLE.GetAtomicNumber(line[0]))
                self.coords.append([float(x.replace('.*^', 'e').replace('*^', 'e')) for x in line[1:]])
            f.readline()    # frequencies
            self.smi = f.readline().split()[0]

    def init_graph(self, Hs_in_fragments=True, Hs_in_linkages=True):
        """
        Decompose the atomic graph of a molecule into a substructure graph. Hydrogen atoms are ignored.
        -------------------------------------------------------------------------------------------
        Returns:
            nx.Graph() object
        """
        graph = nx.Graph()  # initiate graph
        graph.graph["FilePath"] = self.filepath
        graph.graph['Smiles'] = self.smi
        graph.graph['Gap'] = self.gap

        if self.n_atoms < 3:
            _nodes = [list(range(self.n_atoms))]
        else:
            _nodes = [list(x) for x in SSSR(self.mol)]    # Get smallest set of smallest rings (SSSR)
                                                # Note that the definition is not unique
            for bond in self.mol.GetBonds():
                if not bond.IsInRing():     # include the rest of the bonds that are not in a ring
                    _nodes.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        
        graph.add_nodes_from([(i, {'AtomIdxs': atoms}) for i, atoms in enumerate(_nodes)])
        
        for i in graph.nodes:
            # print(f"Node failed\t{self.smi}\t{graph.nodes[i]['AtomIdxs']}")
            _smi, _mol = get_substruct(self.smi, graph.nodes[i]['AtomIdxs'])
            _smi = Chem.MolToSmiles(_mol, kekuleSmiles=True)
            if Hs_in_fragments:
                _mol = Chem.AddHs(_mol)

            self.fragments.append(_smi)
            graph.nodes[i]['Smiles'] = _smi
            graph.nodes[i]['Molecule'] = _mol
            graph.nodes[i]['Atoms'] = [self.atomic_nums[x] for x in graph.nodes[i]['AtomIdxs']]
            graph.nodes[i]['Coordinates'] = [self.coords[x] for x in graph.nodes[i]['AtomIdxs']]
        
        self.n_fragments = len(self.fragments)
        _edges = {}
        for i in range(self.n_fragments)[:-1]:
            a = list(graph.nodes)[i]
            for j in range(self.n_fragments)[i+1:]:
                b = list(graph.nodes)[j]
                shared_atoms = list(set(graph.nodes[a]['AtomIdxs']) & set(graph.nodes[b]['AtomIdxs']))
                if shared_atoms:
                    graph.add_edge(a, b, LinkAtomIdxs=shared_atoms)
                    # print(f"Edge failed\t{self.smi}\t{shared_atoms}")
                    # _smi, _mol = get_substruct(self.smi, shared_atoms)
                    _smi = Chem.MolFragmentToSmiles(self.mol, shared_atoms, kekuleSmiles=True)
                    _mol = Chem.MolFromSmiles(_smi, sanitize=True)
                    _smi = Chem.MolToSmiles(_mol, kekuleSmiles=True)
                    if Hs_in_linkages:
                        _mol = Chem.AddHs(_mol)

                    self.linkages.append(_smi)
                    graph.edges[a, b]['Smiles'] = _smi
                    graph.edges[a, b]['Molecule'] = _mol

        return graph
    
    def update_feature(self, NodeConverter=None, EdgeConverter=None):
        """Extract features from nodes and/or edges or update previous features.

        Args:
            NodeConverter (Function, optional): Node feature extraction function. Defaults to None.
            EdgeConverter (Function, optional): Edge feature extraction function. Defaults to None.
        """
        if isinstance(NodeConverter, types.FunctionType):
            for i in list(self.graph.nodes):
                self.graph[i]['Features'] = NodeConverter(self.graph[i]['Molecule'])
        elif NodeConverter is not None:
            raise Exception("NodeConverter is not a valid converting function.")

        if isinstance(EdgeConverter, types.FunctionType):
            for (a, b, mol) in self.graph.edges.data('Molecule'):
                self.graph.edges[a, b]['Features'] = EdgeConverter(mol)
        elif EdgeConverter is not None:
            raise Exception("EdgeConverter is not a valid converting function.")

def fingerprint_pca(fp_list, n_comps):
    pca = PCA(n_components=n_comps)
    crds = pca.fit_transform(fp_list)
    
    return crds
        

if __name__ == '__main__':
    import os
    print(os.getcwd())

    file_path = "./baselines/data/qm9/xyz/dsgdb9nsd_000608.xyz"
    filenames = [filename for filename in os.listdir("./baselines/data/qm9/xyz/") if filename.endswith(".xyz")]

    all_fps = []
    num_of_fps_per_mol_map = {}
    files_dict = {}
    print(f"{len(filenames)} xyz files")
    for n,file_path in enumerate(filenames):
        G = SubstructGraph("./baselines/data/qm9/xyz/"+file_path)
        node_fp_list = []
        for i in G.graph.nodes:
            node_m = G.graph.nodes[i]['Molecule']
            node_fp = Chem.GetMorganFingerprintAsBitVect(node_m,2,nBits=1024)
            node_fp_list.append(node_fp)
        files_dict[n] = file_path
        num_of_fps_per_mol_map[n] = len(node_fp_list)
        all_fps.append(node_fp_list)
        if n % 100 == 0:
            print(f"{n} xyz files processed")
    print(f"{len(all_fps)} xyz files processed - finished")
    print(f"morgan fingerprints obtained for => {len(all_fps)} xyz files")
    fp_lengths = []
    for i in all_fps:
        fp_lengths.append(len(i))
    stop_indices = (np.cumsum(fp_lengths)).tolist()
    all_fps = [item for sublist in all_fps for item in sublist]
    print(f"{len(all_fps)} total substructures")
    vects = fingerprint_pca(all_fps, 4).tolist()
    print(f"{len(vects)} reduced vectors")
    start_index = 0
    reduced_fps = []
    reduced_fp_dict = {}
    for n, stop_index in enumerate(stop_indices):
        reduced_fp = vects[start_index:stop_index]
        reduced_fp_dict[files_dict[n]] = reduced_fp
        start_index = stop_index
        if num_of_fps_per_mol_map[n] != len(reduced_fp):
            print(f"{files_dict[n]}, expected number of nodes {num_of_fps_per_mol_map[n]}, actual number of nodes {len(reduced_fp)}")
    import json
    with open('reduced_fps.json', 'w') as f:
        json.dump(reduced_fp_dict, f)
    print(f"reduced finger prints obtained for => {len(reduced_fp_dict)} xyz files")

    for file_path in filenames:
        try:
            G = SubstructGraph("./baselines/data/qm9/xyz/"+file_path)
        except:
            print(f"Darn it. Thif file failed: {file_path}")