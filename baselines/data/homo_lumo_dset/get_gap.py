import pandas as pd
import numpy as np
import os

df = pd.read_csv("homo_lumo.csv")

filenames = df["Filename"].tolist()
homo = df["PM7 HOMO(eV)"].tolist()
lumo = df["PM7 LUMO(eV)"].tolist()
gaps = [round(lumo[i]-e,3) for i,e in enumerate(homo)]

data_dict = {}
for index, filename in enumerate(filenames):
    data_dict[filename] = gaps[index]

df_smiles = pd.read_csv("smiles.out", sep="\t", names=['smiles','code'])
smiles = df_smiles["smiles"].tolist()
code = [string[:-4] for string in df_smiles["code"].tolist()]
smiles_dict = {}
for index, smile in enumerate(smiles):
    smiles_dict[code[index]] = smile

print("gap and smiles dict created...")

for subdir, dirs, files in os.walk("xyz_inputs/"):
    filenames = []
    files = files
    file_number = 0
    failed = 0
    for n,f in enumerate(files):
        print(f)
        filename = f[0:-4]
        print(filename)
        """
        try:
            gap = data_dict[filename]
            with open("xyz/"+f) as xyz_file:
                lines = xyz_file.read().splitlines()
                lines[1] = gap
                lines.append(smiles_dict[filename])            
                with open("xyz_inputs/{}.xyz".format(str(file_number).zfill(7)), "w") as mock:
                    for line in lines:
                        if line == lines[0]:
                            mock.write(str(line)+"\n")
                            mock.write(filename+".out"+"\n")
                        elif line == lines[1]:
                            mock.write("gap: "+str(line)+"\n")
                        else:
                            mock.write(str(line)+"\n")
                    mock.write("no frequencies")
            file_number += 1
            if n % 1000 == 0 or (n+1) == len(files):
                print("{} out of {} molecules converted".format(n, len(files)))
        except:
            failed += 1
        """

print("\nsuccess rate = {} ({}/{})".format((file_number)/(file_number + failed), file_number, file_number + failed))
