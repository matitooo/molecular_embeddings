from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from PIL import Image

# Define the SMILES of the molecule.
smiles = 'CC(=O)OC1=CC=CC=CC=C1C(=O)OC1=CC=C1C(=O)O'

# Create the molecule from the SMILES.
molecule = Chem.MolFromSmiles(smiles)

# Draw the molecule.
img = Draw.MolToImage(molecule)
img.save('test.png')

# Calculate the molecular weight.
molecular_weight = Descriptors.MolWt(molecule)
print(f"Mass: {molecular_weight:.3f} g/mol")

name = molecule.GetProp('_Name')