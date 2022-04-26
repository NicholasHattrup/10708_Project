import types
from matplotlib.pyplot import get
import numpy as np
import pandas as pd
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

def get_substruc(mol, atoms):
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
        new_mol = Chem.Kekulize(new_mol.GetMol())
        return new_smi, new_mol
    except:
        raise Warning(f"New molecule creation failed.")
        return None, None

class SubstructGraph(object):
    """
    Substructure graph of a molecule. Hydrogen atoms are ignored when building the graph but may be considered in features.
    """
    def __init__(self, xyz_file, Hs_in_fragments=True, Hs_in_linkages=False):
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

    def init_graph(self, Hs_in_fragments=True, Hs_in_linkages=False):
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
            # try:  
            #     _smi = Chem.MolFragmentToSmiles(self.mol, graph.nodes[i]['AtomIdxs'], kekuleSmiles=True)
            # except:
            #     _smi = Chem.MolFragmentToSmiles(self.mol, graph.nodes[i]['AtomIdxs'], kekuleSmiles=False)
            #     print(f"Node failed\t{_smi}\t{self.filaname}")
            # _mol = Chem.MolFromSmiles(_smi, sanitize=True)
            try:
                _smi, _mol = get_substruc(self.mol, graph.nodes[i]['AtomIdxs'])
            except:
                print(f"Node failed\t{self.smi}\t{graph.nodes[i]['AtomIdxs']}")
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
                    # try:
                    #     _smi = Chem.MolFragmentToSmiles(self.mol, shared_atoms, kekuleSmiles=True)
                    # except:
                    #     _smi = Chem.MolFragmentToSmiles(self.mol, shared_atoms, kekuleSmiles=False)
                    #     print(f"Edge failed\t{_smi}\t{self.filaname}")
                    # _mol = Chem.MolFromSmiles(_smi, sanitize=True)
                    try:
                        _smi, _mol = get_substruc(self.mol, shared_atoms)
                    except:
                        print(f"Edge failed\t{self.smi}\t{shared_atoms}")
                    if Hs_in_linkages:
                        _mol = Chem.AddHs(_mol)

                    self.linkages.append(_smi)
                    graph.edges[a, b]['Smiles'] = _smi
                    graph.edges[a, b]['Molecule'] = _mol
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



if __name__ == '__main__':
    import os
    print(os.getcwd())

    file_path = "./baselines/data/qm9/xyz/dsgdb9nsd_000608.xyz"
    filenames = [filename for filename in os.listdir("./baselines/data/qm9/xyz/") if filename.endswith(".xyz")]
    for file_path in filenames[10:40]:
        G = SubstructGraph("./baselines/data/qm9/xyz/"+file_path)