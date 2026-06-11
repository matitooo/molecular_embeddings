from graph_utils import read_data,return_dicts,smiles_to_batch
from models.graph_encoder import MoleculeGraphEncoder

data_path = 'data/smiles_200.txt'
smiles = read_data(data_path)
dicts = return_dicts()
batch_size = 32

batches = smiles_to_batch(smiles,dicts,batch_size)
model = MoleculeGraphEncoder(node_dim=batches[0].x.shape[1])

for batch in batches:
  model(batch)