import types
from importlib_resources import path
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

 
def get_substruct_node(mol, atoms, add_Hs=True):
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
                if add_Hs:
                    Chem.AddHs(new_mol)
                new_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=False)
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
    new_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=False)
    if add_Hs:
        Chem.AddHs(new_mol)
    return new_smi, new_mol


def get_substruct_edge(mol, atoms, add_Hs=True):
    new_smi = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
    new_mol = Chem.MolFromSmiles(new_smi, sanitize=True)
    new_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=False)
    if add_Hs:
        new_mol = Chem.AddHs(new_mol)
    return new_smi, new_mol


class SubstructGraph(object):
    """
    Substructure graph of a molecule. Hydrogen atoms are ignored when building the graph but may be considered in features.
    """
    def __init__(self, xyz_file, Hs_in_fragments=True, Hs_in_linkages=True):
        self.filepath = xyz_file
        self.filename = xyz_file.split("/")[-1]
        self.parse_xyz_file()  # self.n_atoms, self.smi, self.atoms, self.coord, self.gap
        self.mol = Chem.MolFromSmiles(self.smi)
        self.mol_unsanitized = Chem.MolFromSmiles(self.smi, sanitize=False)
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
                self.coords.append([float(x.replace('.*^', 'e').replace('*^', 'e')) for x in line[1:4]])
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
            _smi, _mol = get_substruct_node(self.mol_unsanitized, graph.nodes[i]['AtomIdxs'], add_Hs=Hs_in_fragments)

            self.fragments.append(_smi)
            graph.nodes[i]['Smiles'] = _smi
            graph.nodes[i]['Molecule'] = _mol
            # graph.nodes[i]['Atoms'] = [self.atomic_nums[x] for x in graph.nodes[i]['AtomIdxs']]
            # graph.nodes[i]['Coords'] = [self.coords[x] for x in graph.nodes[i]['AtomIdxs']]
            weights = [self.atomic_nums[x] for x in graph.nodes[i]['AtomIdxs']]
            coords = np.array([self.coords[x] for x in graph.nodes[i]['AtomIdxs']])
            graph.nodes[i]['WeightedCoords'] = list(np.average(coords, weights=weights, axis=0))
        
        self.n_fragments = len(self.fragments)
        for i in range(self.n_fragments-1):
            a = list(graph.nodes)[i]
            for j in range(i+1, self.n_fragments):
                b = list(graph.nodes)[j]
                shared_atoms = list(set(graph.nodes[a]['AtomIdxs']) & set(graph.nodes[b]['AtomIdxs']))
                if shared_atoms:
                    graph.add_edge(a, b, AtomIdxs=shared_atoms)
                    _smi, _mol = get_substruct_edge(self.mol_unsanitized, shared_atoms, add_Hs=Hs_in_linkages)

                    self.linkages.append(_smi)
                    graph.edges[a, b]['Smiles'] = _smi
                    graph.edges[a, b]['Molecule'] = _mol
                    # graph.edges[a, b]['Atoms'] = [self.atomic_nums[x] for x in graph.edges[a, b]['AtomIdxs']]
                    # graph.edges[a, b]['Coords'] = [self.coords[x] for x in graph.edges[a, b]['AtomIdxs']]
                    # weights = [self.atomic_nums[x] for x in graph.edges[a, b]['AtomIdxs']]
                    # coords = [self.coords[x] for x in graph.edges[a, b]['AtomIdxs']]
                    # graph.edges[a, b]['WeightedCoords'] = list(np.average(coords, weights=weights, axis=0))
        return graph
    
    def update_graph(self, NodeConverter=None, EdgeConverter=None):
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
    
    @staticmethod
    def update_features(graph, NodeConverter=None, EdgeConverter=None):
        """Extract features from nodes and/or edges or update previous features.

        Args:
            g (nx.Graph)
            NodeConverter (Function, optional): Node feature extraction function. Defaults to None.
            EdgeConverter (Function, optional): Edge feature extraction function. Defaults to None.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Input graph is not a valid nx.Graph object.")
            return None
        if isinstance(NodeConverter, types.FunctionType):
            for i in list(graph.nodes):
                graph[i]['Features'] = NodeConverter(graph[i]['Molecule'])
        elif NodeConverter is not None:
            raise Exception("NodeConverter is not a valid converting function.")

        if isinstance(EdgeConverter, types.FunctionType):
            for (a, b, mol) in graph.edges.data('Molecule'):
                graph.edges[a, b]['Features'] = EdgeConverter(mol)
        elif EdgeConverter is not None:
            raise Exception("EdgeConverter is not a valid converting function.")
        
        return graph

def fingerprint_pca(fp_list, n_comps):
    pca = PCA(n_components=n_comps)
    crds = pca.fit_transform(fp_list)
    
    return crds


class GraphLibrary(object):
    """A library collection of substucture graphs.
    """
    def __init__(self, directory="./data/qm9/xyz/", filenames=None):
        if filenames is None:
            filenames = [filename for filename in os.listdir(directory) if filename.endswith(".xyz")]
        self.directory = directory
        self.filenames = filenames

        self.graph_library = []
        for filename in filenames:
            G = SubstructGraph(directory+filename)
            self.graph_library.append(G)
    
    def init_reduction(self):
        self.fragment_library = set([])
        self.linkage_library = set([])
        
        for G in self.graph_library:
            fragments = set(G.fragments)
            linkages = set(G.linkages)
            self.fragment_library = self.fragment_library.union(fragments)
            self.linkage_library = self.linkage_library.union(linkages)



if __name__ == '__main__':
    import os
    print(os.getcwd())

    filepath = "./baselines/data/qm9/xyz/dsgdb9nsd_000608.xyz"
    filenames = [filename for filename in os.listdir("./baselines/data/qm9/xyz/") if filename.endswith(".xyz")]

    all_fps = []
    num_of_fps_per_mol_map = {}
    files_dict = {}
    print(f"{len(filenames)} xyz files")
    for n,filepath in enumerate(filenames):
        G = SubstructGraph("./baselines/data/qm9/xyz/"+filepath)
        node_fp_list = []
        for i in G.graph.nodes:
            node_m = G.graph.nodes[i]['Molecule']
            node_fp = Chem.GetMorganFingerprintAsBitVect(node_m,2,nBits=1024)
            node_fp_list.append(node_fp)
        files_dict[n] = filepath
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
            print(f"Darn it. This file failed: {file_path}")