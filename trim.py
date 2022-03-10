import pandas as pd
import random

df = pd.read_csv('data_file.txt',sep = ' ')
print(df.describe())
std = df.describe().loc['std'][0]
mean = df.describe().loc['mean'][0]

low = mean - 1*std
high = mean + 1*std

semi_smiles = []
semi_gaps = []

smiles = df['smile(canonical)'].tolist()
gaps = df['gap(eV)'].tolist()

for index, gap in enumerate(gaps):
    coin_flip = random.randint(0,1)
    if gap >= 1.0 and gap <= 5.5:
        semi_gaps.append(gap)
        semi_smiles.append(smiles[index])

zipped_semi = list(zip(semi_smiles,semi_gaps))
semi_df = pd.DataFrame(zipped_semi, columns = ['smile(canonical)','gap(eV)'])

print(semi_df)
print(semi_df.describe())

insulator_smiles = []
insulator_gaps = []

for index, gap in enumerate(gaps):
    if gap >= 6.5:
        insulator_gaps.append(gap)
        insulator_smiles.append(smiles[index])

zipped_insulator = list(zip(insulator_smiles,insulator_gaps))
insulator_df = pd.DataFrame(zipped_insulator, columns = ['smile(canonical)','gap(eV)']).sample(n=len(semi_df))

print(insulator_df)
print(insulator_df.describe())
    
