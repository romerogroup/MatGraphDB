import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool

from matgraphdb.pyg.core.model import BaseModel
from matgraphdb.pyg.models.base import InputLayer, MultiLayerPercetronLayer


class CGConvBlock(torch.nn.Module):
    def __init__(self, input_channels, num_edge_features):
        super(CGConvBlock, self).__init__()
        self.num_edge_features = num_edge_features
        self.conv = CGConv(input_channels, dim=num_edge_features)
        self.ffw = MultiLayerPercetronLayer(input_channels, input_channels)

    def forward(self, x, edge_index, edge_attr):
        if self.num_edge_features > 0:
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            x = self.conv(x=x, edge_index=edge_index)
        x = self.ffw(x)
        return x


class CGConvModel(BaseModel):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        out_channels,
        num_layers=2,
        num_ffw_layers=1,
        pool_fn=global_mean_pool,
        task_type: str = "regression",
    ):
        super(CGConvModel, self).__init__(task_type=task_type)
        self.pool_fn = pool_fn
        self.num_layers = num_layers
        self.num_edge_features = num_edge_features
        self.input_layer = InputLayer(num_node_features, hidden_channels)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(CGConvBlock(hidden_channels, num_edge_features))

        self.ffw_layers = nn.ModuleList()
        for i in range(num_ffw_layers):
            self.ffw_layers.append(MultiLayerPercetronLayer(hidden_channels, 1))

        # A final linear layer for downstream prediction
        self.output_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch = data.batch if hasattr(data, "batch") else None

        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Global pooling (mean pooling as an example)
        if batch is not None:
            x = self.pool_fn(x, batch)

        for layer in self.ffw_layers:
            x = layer(x)

        # Final linear layer
        x = self.output_layer(x)

        return x

    def compute_loss(self, data, outputs, loss_fn):
        return loss_fn(outputs, data.y)


class CGConvModel(BaseModel):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        out_channels,
        num_layers=2,
        num_ffw_layers=1,
        pool_fn=global_mean_pool,
        task_type: str = "regression",  # 'regression', 'binary', 'multiclass'
    ):
        super().__init__()
        self.task_type = task_type

        # Validate task configuration
        self._validate_task_config(out_channels)

        # Existing architecture components
        self.pool_fn = pool_fn
        self.num_layers = num_layers
        self.num_edge_features = num_edge_features
        self.input_layer = InputLayer(num_node_features, hidden_channels)

        # Message passing blocks
        self.blocks = nn.ModuleList(
            [CGConvBlock(hidden_channels, num_edge_features) for _ in range(num_layers)]
        )

        # Feedforward layers
        self.ffw_layers = nn.ModuleList(
            [
                MultiLayerPercetronLayer(hidden_channels, hidden_channels)
                for _ in range(num_ffw_layers)
            ]
        )

        # Final output layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)

        # Optional activation (can be none for logits)
        self.output_activation = self._get_output_activation()

    def _validate_task_config(self, out_channels):
        """Ensure valid combination of task type and output channels"""
        if self.task_type == "binary" and out_channels != 1:
            raise ValueError("Binary classification requires out_channels=1")
        if self.task_type == "multiclass" and out_channels < 2:
            raise ValueError("Multiclass requires out_channels >= 2")

    def _get_output_activation(self):
        """Get appropriate activation for task type"""
        return {
            "regression": nn.Identity(),
            "binary": nn.Sigmoid(),
            "multiclass": nn.Identity(),  # CrossEntropy handles activation
        }[self.task_type]

    def forward(self, data):
        # Existing forward logic
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)

        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if batch is not None:
            x = self.pool_fn(x, batch)

        for layer in self.ffw_layers:
            x = layer(x)

        # Apply final output transformations
        return self.output_activation(self.output_layer(x))

    def compute_loss(self, data, outputs, loss_fn=None):
        """
        Handles different task types with automatic target reshaping
        and default loss function selection
        """
        # Get default loss if not provided
        if loss_fn is None:
            loss_fn = self._get_default_loss_fn()

        # Process targets based on task type
        targets = self._process_targets(data.y)

        # print(f"targets: {targets.shape}")
        # print(f"outputs: {outputs.shape}")
        return loss_fn(outputs, targets)

    def _get_default_loss_fn(self):
        """Select appropriate default loss based on task type"""
        return {
            "regression": nn.MSELoss(),
            "binary": nn.BCELoss(),  # Use with Sigmoid output
            "multiclass": nn.CrossEntropyLoss(),
        }[self.task_type]

    def _process_targets(self, targets):
        """Ensure proper target formatting for different tasks"""
        if self.task_type == "multiclass":
            return targets.long().squeeze()  # Class indices

        # For regression/binary: ensure float32 and matching shape
        targets = targets.float().view(-1, 1)
        if self.task_type == "binary":
            return targets.view(-1, 1)  # BCELoss expects same shape as output
        return targets
