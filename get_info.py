import os
import pathlib
smiles = []
gaps = []
for xyz_file in pathlib.Path("xyz_files").glob('*xyz'):
    list_of_lines = []
    with open("{}".format(xyz_file),"r") as f:
        for line in f:
            line.strip()
            linelist = line.split()
            list_of_lines.append(linelist)
        smilesL = list_of_lines[-2]
        paramL = list_of_lines[1]
        smiles.append(smilesL[1])
        gaps.append(paramL[9])
        
gaps = [float(gap)*27.2114 for gap in gaps]

with open('data_file.txt','w') as f:
    f.write('smile(canonical) gap(eV)\n')
    for index, value in enumerate(smiles):
        f.write('{} {}\n'.format(value, gaps[index]))

    
