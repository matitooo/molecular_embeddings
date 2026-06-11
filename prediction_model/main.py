from functools import partial
from datasets import DropArray
from utils import batch_instances_graph
from model import MoleculeGraphEncoder,DrugCombinationModel
import torch
from train_utils import train_loop

print('Loading Dataset and Vectorizing Molecules')
dataset = DropArray("data/droparray_100.pt")

collate_fn = partial(batch_instances_graph, drug_graph_dict=dataset.drug_graph_dict)

train, test = dataset.get_split(how="new_drugs", fold=0)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=128, num_workers=0,
    collate_fn=collate_fn, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=128, num_workers=0,
    collate_fn=collate_fn, shuffle=True
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 128
LR           = 1e-3
EPOCHS       = 50
EMBEDDING_DIM = 128
HIDDEN_DIM   = 256
FOLD         = 0

mol_encoder = MoleculeGraphEncoder(
    node_dim=47,
    hidden_dim=HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_layers=4
)


model = DrugCombinationModel(
    mol_encoder=mol_encoder,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_loop(model,optimizer,DEVICE,train_loader,test_loader,EPOCHS)