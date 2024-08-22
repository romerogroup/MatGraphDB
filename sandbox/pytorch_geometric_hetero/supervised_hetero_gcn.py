# Creating a GraphSAGE model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import  SAGEConv, GCNConv, GraphConv, to_hetero


from matgraphdb.graph_kit.pyg.models import MultiLayerPerceptron
from matgraphdb.graph_kit.data import DataGenerator
from matgraphdb.graph_kit.pyg.metrics import ClassificationMetrics,RegressionMetrics

class StackedGraphConvLayers(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, 
                normalize=False, dropout=0.2 ):
        super(StackedGraphConvLayers, self).__init__()
        self.dropout=dropout
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            GraphConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 2)
        ])
        

    def forward(self, x: torch.Tensor, edge_index:  torch.Tensor,training=False):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout,training=training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=self.dropout,training=training)
        
        return x

class SupervisedHeteroSAGEModel(nn.Module):
    def __init__(self, data,
                 hidden_channels:int, 
                 out_channels:int,
                 prediction_node_type:str,
                 num_layers,
                 device='cuda:0'):
        super(SupervisedHeteroSAGEModel, self).__init__()
        self.embs = nn.ModuleDict()
        self.data_lins = nn.ModuleDict()
        self.prediction_node_type=prediction_node_type
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            num_features = data[node_type].num_node_features
            self.embs[node_type]=nn.Embedding(num_nodes,hidden_channels,device=device)
            if num_features != 0:
                self.data_lins[node_type]=nn.Linear(num_features, hidden_channels,device=device)

        # self.output_layer = MultiLayerPerceptron(input_dim=hidden_channels,
        #                                 output_dim=out_channels,
        #                                 num_layers=1,
        #                                 n_embd=hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
        # Initialize and convert GraphSAGE to heterogeneous
        self.graph_sage = StackedGraphConvLayers(hidden_channels,hidden_channels,num_layers)
        self.graph_sage = to_hetero(self.graph_sage, metadata=data.metadata())

    def forward(self, data):
        x_dict={}
        edge_attr_dict={}
        for node_type, emb_layer in self.embs.items():
            # Handling nodes based on feature availability
            if node_type in self.data_lins:
                x_dict[node_type] = self.data_lins[node_type](data[node_type].x) + emb_layer(data[node_type].node_id)
            else:
                x_dict[node_type] = emb_layer(data[node_type].node_id)


        edge_attr_dict[node_type] = data[node_type].edge_attr

        x_dict=self.graph_sage(x_dict, edge_attr_dict)

        out=self.output_layer(x_dict[self.prediction_node_type] + self.data_lins[self.prediction_node_type](data[self.prediction_node_type].x))
        return out
    




# rev_edge_types=[]
# edge_types=[]
# for edge_type in graph_dataset.data.edge_types:
#     rel_type=edge_type[1]
#     if 'rev' in rel_type:
#         rev_edge_types.append(edge_type)
#     else:
#         edge_types.append(edge_type)
# transform = T.RandomLinkSplit(
#         num_val=0.1,
#         num_test=0.1,
#         disjoint_train_ratio=0.3,
#         neg_sampling_ratio=2.0,
#         add_negative_train_samples=False,
#         edge_types=edge_types,
#         rev_edge_types=rev_edge_types, 
#     )

# transform = T.RandomNodeSplit(split="random")  # Or another appropriate method


def get_node_dataloaders(data,node_type,shuffle=False):
    data[node_type].node_id
    input_nodes=('material',data[node_type].node_id)
    loader = NeighborLoader(
                graph_dataset.data,
                # Sample 15 neighbors for each node and each edge type for 2 iterations:
                num_neighbors=[15] * 2,
                replace=False,
                subgraph_type="bidirectional",
                disjoint=False,
                weight_attr = None,
                transform=None,
                transform_sampler_output = None,
                
                input_nodes=input_nodes,
                shuffle=shuffle,
                batch_size=128,
            )
    return loader


def split_data_on_node_type(data,node_type,train_proportion=0.8,test_proportion=0.1, val_proportion=0.1):
    assert train_proportion + test_proportion + val_proportion == 1.0
    for node_type in data.node_types:
        train_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)
        test_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)
        val_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)

        num_nodes_for_type=data[node_type].num_nodes
        if node_type==NODE_TYPE:
            # Determine indices for training, testing, and validation
            indices = torch.randperm(num_nodes_for_type)

            num_train = int(train_proportion * num_nodes_for_type)
            num_val = int(test_proportion * num_nodes_for_type)
            num_test = num_nodes_for_type - num_train - num_val

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train + num_val]] = True
            test_mask[indices[num_train + num_val:]] = True
        else:
            train_mask[:num_nodes_for_type]=True

        data[node_type].train_mask=train_mask
        data[node_type].test_mask=test_mask
        data[node_type].val_mask=val_mask
    return data






NODE_TYPE='material'
# 
# TARGET_PROPERTY='energy_above_hull'
# TARGET_PROPERTY='formation_energy_per_atom'
# TARGET_PROPERTY='energy_per_atom'
# TARGET_PROPERTY='band_gap'
TARGET_PROPERTY='k_vrh'
# TARGET_PROPERTY='density'
# TARGET_PROPERTY='density_atomic'

# TARGET_PROPERTY='crystal_system'
# TARGET_PROPERTY='point_group'
# TARGET_PROPERTY='nelements'
# TARGET_PROPERTY='elements'

CONNECTION_TYPE='GEOMETRIC_ELECTRIC_CONNECTS'
# CONNECTION_TYPE='GEOMETRIC_CONNECTS'
# CONNECTION_TYPE='ELECTRIC_CONNECTS'

# Training params
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.1
VAL_PROPORTION = 0.1
LEARNING_RATE = 0.001
N_EPCOHS = 1000

# model params

NUM_LAYERS = 2
HIDDEN_CHANNELS = 128
EVAL_INTERVAL = 10

properties_per_node_type={
    'element':['atomic_number','group','row','atomic_mass'],
    'material':[
        'composition',
        # 'space_group',
        # 'nelements',
        # 'nsites',
        # 'crystal_system',
        # 'density',
        # 'density_atomic',
        # 'volume',
        # 'g_vrh',
        ],
}
properties=[]
for node_type in properties_per_node_type.keys():
    properties.extend(properties_per_node_type[node_type])



data=graph_dataset.data
OUT_CHANNELS=data[NODE_TYPE].out_channels


# transform = T.RandomNodeSplit(
#                         split="random",
#                         num_splits=1,
#                         split="random",
#                         )
# train_mask=torch.zeros(graph_dataset.data.num_nodes,dtype=torch.bool)
# test_mask=torch.zeros(graph_dataset.data.num_nodes,dtype=torch.bool)
# val_mask=torch.zeros(graph_dataset.data.num_nodes,dtype=torch.bool)

device =  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
data=split_data_on_node_type(data,
                            node_type=NODE_TYPE,
                            train_proportion=TRAIN_PROPORTION,
                            test_proportion=TEST_PROPORTION,
                            val_proportion=VAL_PROPORTION)

print(data)
print(data.edge_types)


model = SupervisedHeteroSAGEModel(data,
                            hidden_channels=HIDDEN_CHANNELS, 
                            out_channels=OUT_CHANNELS,
                            prediction_node_type=NODE_TYPE,
                            num_layers=NUM_LAYERS,
                            device=device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if OUT_CHANNELS==1:
    loss_fn=nn.MSELoss()
else:
    loss_fn=nn.CrossEntropyLoss()


def train(model, optimizer, data, loss_fn):
    model.train()
    total_loss = 0
    total_examples = 0

    data = data.to(device)
    optimizer.zero_grad()

    out = model(data)
    mask=data[NODE_TYPE].train_mask
    pred=out[mask]
    ground_truth=data[NODE_TYPE].y[mask]

    loss = loss_fn(pred, ground_truth)

    loss.backward()
    optimizer.step()
    total_loss += float(loss)
    total_examples += pred.numel()

    return total_loss 


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        
        losses=[]
        metrics={"val":{},"test":{}}
        data = data.to(device)
        optimizer.zero_grad()

        logits = model(data)
        out_channel=logits.shape[1]
        for key in metrics.keys():
            if out_channel==1:
                metrics[key]['mape']=[]
                metrics[key]['mae']=[]
            else:
                metrics[key]['accuracy']=[]
                metrics[key]['precision']=[]
                metrics[key]['recall']=[]
                metrics[key]['f1']=[]

        for split in ['val_mask', 'test_mask']:
            mask=data[NODE_TYPE][split]
            masked_logits=logits[mask]
            ground_truth=data[NODE_TYPE].y[mask]

            loss = loss_fn(masked_logits, ground_truth)

            out_channel=logits.shape[1]
            split_name=split.split('_')[0]
            if out_channel==1:
                mape=RegressionMetrics.mean_absolute_percentage_error(y_pred=masked_logits,y_true=ground_truth)
                mae=RegressionMetrics.mean_absolute_error(y_pred=masked_logits,y_true=ground_truth)
                metrics[split_name]['mape'].append(mape.item())
                metrics[split_name]['mae'].append(mae.item())
            else:
                probabilities = torch.sigmoid(masked_logits)
                # Converting masked_logits from (batch_size, out_channels) to (batch_size,)
                pred=probabilities.argmax(1)

                
                # accuracy=ClassificationMetrics.accuracy(y_pred=pred,y_true=ground_truth)
                # metrics[split_name]['accuracy'].append(accuracy.item()*100)

                cm=ClassificationMetrics.confusion_matrix(y_pred=pred,y_true=ground_truth,num_classes=out_channel)

                weights=cm.sum(dim=1)/cm.sum(dim=1).sum()

                accuracy=ClassificationMetrics.multi_class_accuracy(confusion_matrix=cm)
                avg_accuracy=(weights * accuracy).sum()
                metrics[split_name]['accuracy'].append(avg_accuracy*100)

                precision=ClassificationMetrics.multiclass_precision(confusion_matrix=cm)
                avg_precision= (weights * precision).sum()
                metrics[split_name]['precision'].append(avg_precision*100)

                recall=ClassificationMetrics.multiclass_recall(confusion_matrix=cm)
                avg_recall=(weights * recall).sum()
                metrics[split_name]['recall'].append(avg_recall*100)

                f1=ClassificationMetrics.multiclass_f1_score(confusion_matrix=cm)
                avg_f1=(weights * f1).sum()
                metrics[split_name]['f1'].append(avg_f1*100)

            losses.append(loss.item())

    return losses,metrics



for epoch in range(N_EPCOHS):
    train_loss = train(model, optimizer, data, loss_fn=loss_fn)
    if epoch%EVAL_INTERVAL==0:
        losses,metrics = evaluate(model, data)

        # print(f"Epoch: {epoch:03d},Train Loss: {train_loss:.4f}, Val Loss: {losses[0]:.4f}, Test Loss: {losses[1]:.4f}")

        metrics_str=""
        metrics_str+=f"Epoch: {epoch:03d},Train Loss: {train_loss:.4f}, Test Loss: {losses[1]:.4f}"
        for split,metrics_dict in metrics.items():
            metrics_str+=" | "
            for i,(key,value) in enumerate(metrics_dict.items()):
                if i==0:
                    metrics_str+=f" {split}-{key}: {value[0]:.2f}"
                else:
                    metrics_str+=f", {split}-{key}: {value[0]:.2f}"
        print(metrics_str)