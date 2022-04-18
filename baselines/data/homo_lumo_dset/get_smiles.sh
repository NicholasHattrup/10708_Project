for smi in xyz/*
do
    obabel -ixyz $smi -osmi >> smiles.out
done

    
