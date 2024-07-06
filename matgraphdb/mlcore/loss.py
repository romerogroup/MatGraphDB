import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import negative_sampling
from torch_cluster import random_walk
import torch.nn.functional as F

import random

def negative_sampling_loss(z, edge_index, num_neg_samples=10):
    src, pos = edge_index[0], edge_index[1]
    neg = torch.randint(0, z.size(0), (num_neg_samples * src.size(0),), dtype=torch.long)

    pos_loss = -torch.log(torch.sigmoid((z[src] * z[pos]).sum(dim=-1))).mean()
    neg_loss = -torch.log(1 - torch.sigmoid((z[src] * z[neg]).sum(dim=-1))).mean()
    
    return pos_loss + neg_loss


def positive_sampling_hetero_random_walk_loss(z_dict, edge_index_dict, num_walks=10, walk_length=3, weighted=True,device=None):
    type_loss=torch.zeros(size=(len(z_dict),),device=device)

    for i_node_type,(node_type, z_src) in enumerate(z_dict.items()):
        for node in z_src[:1,:]:
            

            embd_size=node.shape[0]
            neighbors=torch.zeros(size=(num_walks,embd_size),device=device)
            for i_walk in range(num_walks):
                # Resetting the current node type and node to initial node
                current_node_type=node_type
                current_node=node

                # Iterating over the walk length
                for i_step in range(walk_length):
                    edge_types=[]
                    weights=[]
                    for edge_type, edge_index in edge_index_dict.items():
                        src_type, rel_type, dst_type = edge_type[0],edge_type[1], edge_type[2]
                        if current_node_type == src_type:
                            edge_types.append(edge_type)
                            weights.append(edge_index.shape[1])

                    # Randomly selecting the step type
                    if weighted:
                        weights = torch.tensor(weights, dtype=torch.float)
                    else:
                        weights = torch.ones(len(weights), dtype=torch.float)
                    random_step_type_index = torch.multinomial(weights, 1).item()
                    step_edge_type = edge_types[random_step_type_index]
                    src_type, rel_type, dst_type = step_edge_type[0],step_edge_type[1], step_edge_type[2]
                    step_edge_index = edge_index_dict[step_edge_type]

                
                    # Randomly selecting the step from the selected edge type
                    random_step_weights=torch.ones(size=(step_edge_index.shape[1],))
                    random_step_index = torch.multinomial(random_step_weights, 1).item()
                    src_step_index,dst_step_index=step_edge_index[:,random_step_index]

                    current_node_type=dst_type
                    current_node=z_dict[current_node_type][dst_step_index]

                neighbors[i_walk,:]=current_node

            matmul=torch.matmul(node,neighbors.T)
            matmul = torch.nan_to_num(matmul, nan=0.0)

            type_loss[i_node_type] += -torch.log(torch.sigmoid( matmul.mean() ) )
    pos_loss=type_loss.sum()
    return pos_loss



def negative_sampling_hetero_loss(z_dict, edge_index_dict, num_neg_samples=10, method = "sparse"):
    total_loss = 0

    for node_type, z_src in z_dict.items():
        # pos_batch = random_walk(row, col, batch, 
        #                   walk_length=1,
        #                   coalesced=False)[:, 1]
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type[0],edge_type[1], edge_type[2]
            if node_type == src_type:
                pos_src, pos_dst = edge_index[0], edge_index[1]
                # print(pos_src.shape)

        print(z_src.shape)

        pass
    # for edge_type, edge_index in edge_index_dict.items():
    #     pos_src, pos_dst = edge_index[0], edge_index[1]
    #     src_type, rel_type, dst_type = edge_type[0],edge_type[1], edge_type[2]
    #     # src_z, dst_z = z_dict[src_type][pos_src], z_dict[dst_type][pos_dst]
    #     # print(src_z.shape)
    #     # print(dst_z.shape)
    #     # torch.einsum('aj,bj->i', src_z, dst_z)
    #     # result=src_z* dst_z
    #     # print(result.shape)
    #     neg_sample_edge_index=negative_sampling(
    #         edge_index=edge_index,
    #         # num_nodes = test_data['material', 'has', 'element'].num_nodes,
    #         num_neg_samples = num_neg_samples,
    #         # method = method,
    #         # force_undirected = True,
    #     )
    #     neg_dst, neg_src = neg_sample_edge_index[0], neg_sample_edge_index[1]
    #     neg_src_z, neg_dst_z = z_dict[src_type][neg_src], z_dict[dst_type][neg_dst]
    #     print(neg_dst_z.shape)


        # neg_dst = torch.randint(0, dst_z.size(0), (num_neg_samples * src.size(0),), dtype=torch.long)

        # pos_loss = -torch.log(torch.sigmoid((src_z * dst_z).sum(dim=-1))).mean()
        # neg_loss = -torch.log(1 - torch.sigmoid((src_z * dst_z[neg_dst]).sum(dim=-1))).mean()
        # total_loss += pos_loss + neg_loss
    # return total_loss / len(edge_index_dict)