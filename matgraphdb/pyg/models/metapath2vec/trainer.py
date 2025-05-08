import os

import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from torch_cluster import random_walk
from matgraphdb.pyg.models.hetero_encoder.metrics import (
    LearningCurve,
    ROCCurve,
    plot_pca,
)
from matgraphdb.pyg.models.trainer import BaseTrainer, callbacks


class Trainer(BaseTrainer):

    def unsupervised_graph_loss(self, z, pos_edge_index, random_walk_length=10, Q=1, eps=1e-15):
        r"""Computes the unsupervised graph loss:
        
        .. math::
            \mathcal{L}(z_u) = - \log \sigma(z_u^\top z_v)
            - Q \cdot \mathbb{E}_{v_n\sim P_n}\Big[ \log \sigma\Big(- z_u^\top z_{v_n}\Big)\Big]
        
        for each positive pair \((u,v)\) specified in `pos_edge_index`, where the negative
        samples \(v_n\) are drawn uniformly from the set of nodes.
        
        Args:
            z (Tensor): The node representations of shape
                \([{\tt num\_nodes}, {\tt embedding\_dim}]\).
            pos_edge_index (LongTensor): A tensor of shape \([2, {\tt num\_pos}]\) holding
                the indices of positive node pairs. Each column is a pair \((u, v)\) where
                \(v\) is a node that co-occurs with \(u\) in a fixed-length random walk.
            Q (int, optional): The number of negative samples per positive pair.
                (default: 1)
            eps (float, optional): A small value for numerical stability.
                (default: 1e-15)
        
        Returns:
            loss (Tensor): A scalar loss.
        """
        # --- Positive term ---
        # Extract positive node pairs:
        
        
        u, v = pos_edge_index  # each has shape [num_pos]
        num_nodes = z.size(0)
        
        # Define the starting nodes for random walks.
        # Here we use all nodes as starting points, but you could sample a subset.
        start = torch.arange(num_nodes, device=z.device)
        
        # Sample random walks.
        # node_seq will be of shape [num_walks, walk_length + 1], where num_walks == num_nodes.
        node_seq = random_walk(u, v, start, walk_length=random_walk_length, num_nodes=num_nodes, p=0.5, q=2)
        
        
        # --- Generate Positive Pairs ---
        # For each random walk, treat the starting node (first element) as the anchor,
        # and all subsequent nodes in the walk as positive context nodes.
        # For example, if node_seq[i] = [u, v1, v2, ..., v_walk_length],
        # we create positive pairs: (u, v1), (u, v2), ..., (u, v_walk_length).
        anchor = node_seq[:, 0]        # Shape: [num_walks]
        context = node_seq[:, 1:]        # Shape: [num_walks, walk_length]
        
        # Repeat the anchor for each positive sample in its corresponding walk.
        num_walks, L = node_seq.size()  # L = walk_length + 1
        pos_u = anchor.unsqueeze(1).repeat(1, L - 1).flatten()  # Shape: [num_walks * walk_length]
        pos_v = context.flatten()                               # Shape: [num_walks * walk_length]
        
        # Stack to form pos_edge_index of shape [2, num_pos_pairs]
        pos_edge_index = torch.stack([pos_u, pos_v], dim=0)
        

        u = pos_edge_index[0,:]
        v = pos_edge_index[1,:]
        
        # Dot product between the representations of u and v:
        
        pos_dot = (z[u] * z[v]).sum(dim=-1)  # shape: [num_pos]
        
        # Positive loss: -log σ(z_u^T z_v)
        pos_loss = - torch.log(torch.sigmoid(pos_dot) + eps)
        
        # --- Negative term ---
        num_pos = u.size(0)
        # For each positive pair, sample Q negative nodes (from all nodes)
        neg_sample_idx = torch.randint(0, z.size(0), (num_pos, Q), device=z.device)
        
        # We use the representation of u for each pair and compute its dot product with
        # each of its negative samples.
        z_u = z[u].unsqueeze(1)           # shape: [num_pos, 1, embedding_dim]
        z_neg = z[neg_sample_idx]           # shape: [num_pos, Q, embedding_dim]
        neg_dot = (z_u * z_neg).sum(dim=-1)  # shape: [num_pos, Q]
        
        # Negative loss: - Q * (mean over Q samples of log σ(- z_u^T z_neg) )
        # Equivalently, since expectation is estimated by the sample average,
        # multiplying by Q recovers the sum over the Q samples.
        neg_loss = - Q * torch.log(torch.sigmoid(-neg_dot) + eps).mean(dim=1)

        
        # Total loss: average over all positive pairs
        loss = (pos_loss + neg_loss).mean()

        return loss
    
    
    def train_step(self, data_batch, **kwargs) -> float:
        """
        Performs a validation step.

        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Iterate over self.train_loader.
            - Compute the model output and loss using self.loss_fn.
            - Perform backpropagation and an optimizer step using self.optimizer.
            - Return an aggregated training loss (e.g., the average loss over the epoch).
        Returns:
            float: The training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        data = self.model(data_batch)
        loss = self.unsupervised_graph_loss(data.z, data.edge_index)

        loss.backward()

        self.optimizer.step()
        return loss

    def validation_step(self, data_batch, **kwargs):
        """
        Performs a validation step.
        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Use the provided data_batch from a validation DataLoader.
            - Compute the model output and loss (without performing backpropagation).
            - Return an aggregated validation loss.
        Returns:
            float: The validation loss.
            torch.Tensor: The validation output.
            torch.Tensor: The validation target.
        """
        self.model.eval()
        data = self.model(data_batch)
        loss = self.unsupervised_graph_loss(data.z, data.edge_index)
        return float(loss.cpu().float()), data.z, data.edge_index

    def eval_metrics(self, preds, targets, **kwargs):
        return {}


@callbacks
def learning_curve(trainer):
    learning_curve = LearningCurve()

    for split_label in trainer.split_names:
        epochs = trainer.data_splits_metrics[split_label]["epochs"]
        loss_values = trainer.data_splits_metrics[split_label]["loss"]

        # Add main model curve
        learning_curve.add_curve(
            epochs, loss_values, split_label, split_label, is_baseline=False
        )

    # Generate and save plots
    learning_curve.plot()
    learning_curve.save(os.path.join(trainer.epoch_dir, "learning_curve.png"))
    learning_curve.save(os.path.join(trainer.run_dir, "learning_curve_current.png"))
    learning_curve.close()


@callbacks
def roc_curve(trainer):
    roc_curve_plot = ROCCurve()

    for split_label in trainer.split_names:
        pred = trainer.current_preds
        target = trainer.current_targets

        # Add main model curve
        roc_curve_plot.add_curve(
            pred, target, split_label, split_label, is_baseline=False
        )

    roc_curve_plot.plot()
    roc_curve_plot.save(os.path.join(trainer.epoch_dir, "roc_curve.png"))
    roc_curve_plot.save(os.path.join(trainer.run_dir, "roc_curve_current.png"))
    roc_curve_plot.close()




@callbacks
def pca_plots(trainer):
    pca_dir = os.path.join(trainer.run_dir, "pca")

    for data_split in trainer.split_names:
        split_dir = os.path.join(pca_dir, data_split)
        os.makedirs(split_dir, exist_ok=True)

        trainer.model.eval()

        data = trainer.data_splits[data_split]
        
        n_nodes_per_type = {}
        for node_type in data.node_types:
            n_nodes_per_type[node_type] = data[node_type].num_nodes
        
        # 1. Extract embeddings from the trained model
        with torch.no_grad():
            z = trainer.model.encode(data)

        # 3. Combine embeddings only for the selected node types
        z_all = z.cpu().numpy()
        plot_pca(
            z_all,
            split_dir,
            n_components=4,
            n_nodes_per_type=n_nodes_per_type,
        )


