from graph_utils import read_data,smiles_to_data,return_dicts
from tqdm import tqdm
smiles = read_data()
dicts = return_dicts()
out = []
config = sorted(['chirality','symbol'])

for smile in tqdm(smiles):
  out.append(smiles_to_data(smile,dicts,config))


