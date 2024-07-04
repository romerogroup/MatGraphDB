

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import torch_geometric.transforms as T

import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score

from matgraphdb.mlcore.datasets import load_node_csv, load_edge_csv
from matgraphdb.mlcore.encoders import CategoricalEncoder,IdentityEncoder






class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: torch.Tensor, x_movie: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred



if __name__ == "__main__":
 
    # movie_path = 'data/movies/movies.csv'
    # movie_x, movie_mapping = load_node_csv(
    #     movie_path, index_col='movieId', encoders={
    #         'genres': CategoricalEncoder()
    #     })



    # rating_path = 'data/movies/ratings.csv'
    # _, user_mapping = load_node_csv(rating_path, index_col='userId')

    # data = HeteroData()

    # data["user"].node_id = torch.arange(len(user_mapping))
    # data["movie"].node_id = torch.arange(len(movie_mapping))
    # data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
    # data['movie'].x = movie_x

    # print(data)

    # edge_index, edge_label = load_edge_csv(
    #     rating_path,
    #     src_index_col='userId',
    #     src_mapping=user_mapping,
    #     dst_index_col='movieId',
    #     dst_mapping=movie_mapping,
    #     encoders={'rating': IdentityEncoder(dtype=torch.long)},
    # )

    # data['user', 'rates', 'movie'].edge_index = edge_index
    # data['user', 'rates', 'movie'].edge_label = edge_label

    # # We can leverage the `T.ToUndirected()` transform for this from PyG:
    # data = T.ToUndirected()(data)

    # print(data)

    # Load the entire ratings data frame into memory:

    # Load the entire movie data frame into memory:
    movie_path = 'data/movies/movies.csv'
    movies_df = pd.read_csv(movie_path, index_col='movieId')

    # Split genres and convert into indicator variables:
    genres = movies_df['genres'].str.get_dummies('|')
    print(genres[["Action", "Adventure", "Drama", "Horror"]].head())
    # Use genres as movie input features:
    movie_feat = torch.from_numpy(genres.values).to(torch.float)
    assert movie_feat.size() == (9742, 20)  # 20 genres in total.


    ratings_path = 'data/movies/ratings.csv'
    
    ratings_df = pd.read_csv(ratings_path)

    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()
    # Create a mapping from unique movie indices to range [0, num_movie_nodes):
    unique_movie_id = ratings_df['movieId'].unique()
    unique_movie_id = pd.DataFrame(data={
        'movieId': unique_movie_id,
        'mappedID': pd.RangeIndex(len(unique_movie_id)),
    })

    print("Mapping of movie IDs to consecutive values:")
    print("===========================================")
    print(unique_movie_id.head())
    # Perform merge to obtain the edges from users and movies:
    ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                                left_on='userId', right_on='userId', how='left')
    ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
    ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                                left_on='movieId', right_on='movieId', how='left')
    ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
    assert edge_index_user_to_movie.size() == (2, 100836)
    print()
    print("Final edge indices pointing from users to movies:")
    print("=================================================")
    print(edge_index_user_to_movie)


    data = HeteroData()
    # Save node indices:
    data["user"].node_id = torch.arange(len(unique_user_id))
    data["movie"].node_id = torch.arange(len(movies_df))
    # Add the node features and edge indices:
    data["movie"].x = movie_feat
    data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)



    """
    
    Defining Edge-level Training Splits
    Since our data is now ready-to-be-used, we can split the ratings of users into 
    training, validation, and test splits. 
    This is needed in order to ensure that we leak no information 
    about edges used during evaluation into the training phase. 
    For this, we make use of the transforms.RandomLinkSplit transformation from PyG. 
    This transforms randomly divides the edges in the ("user", "rates", "movie") into training, validation and test edges. 
    The disjoint_train_ratio parameter further separates edges 
    in the training split into edges used for message passing (edge_index) 
    and edges used for supervision (edge_label_index). 
    Note that we also need to specify the reverse edge type ("movie", "rev_rates", "user"). 
    This allows the RandomLinkSplit transform to drop reverse edges accordingly to not leak any information into the training phase.
    
    """
    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    # We can leverage the `RandomLinkSplit()` transform for this from PyG:
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "movie"),
        rev_edge_types=("movie", "rev_rates", "user"), 
    )
    train_data, val_data, test_data = transform(data)



    """
    Defining Mini-batch Loaders
    In this step, we create a mini-batch loader that will generate subgraphs that can be used as input into our GNN. 
    While this step is not strictly necessary for small-scale graphs, 
    it is absolutely necessary to apply GNNs on larger graphs that do not fit onto GPU memory otherwise. 
    Here, we make use of the loader.LinkNeighborLoader which samples multiple hops from
    both ends of a link and creates a subgraph from it. 
    Here, edge_label_index serves as the "seed links" to start sampling from.
    
    """
    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the `loader.LinkNeighborLoader` from PyG:
    from torch_geometric.loader import LinkNeighborLoader

    # Define seed edges:
    edge_label_index = train_data["user", "rates", "movie"].edge_label_index
    edge_label = train_data["user", "rates", "movie"].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )



    

    model = Model(hidden_channels=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["user", "rates", "movie"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")



    # Define the validation seed edges:
    edge_label_index = val_data["user", "rates", "movie"].edge_label_index
    edge_label = val_data["user", "rates", "movie"].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )
    sampled_data = next(iter(val_loader))


    
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")