import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaseModel(torch.nn.Module):
    def __init__(self, name=None, logging=False):
        super(BaseModel, self).__init__()
        self.name = name or self.__class__.__name__.lower()
        self.logging = logging
        self.layers = torch.nn.ModuleList()
        self.loss_value = 0.0
        self.accuracy_value = 0.0

    def _build(self):
        """To be implemented by subclasses: 
           should append layers to self.layers."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Executes the built layers in sequence."""
        if not hasattr(self, '_built') or not self._built:
            self._build()
            self._built = True

        x, edge_index = args
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def loss(self, logits, labels, mask):
        """Masked softmax cross‐entropy + optional weight decay."""
        raise NotImplementedError

    def accuracy(self, logits, labels, mask):
        """Masked accuracy."""
        pred = logits[mask].argmax(dim=1)
        return (pred == labels[mask]).float().mean()


class GCN(BaseModel):
    def __init__(self, placeholders, input_dim, hidden_dim, **kwargs):
        """
        placeholders: dict with keys
          - 'support'       unused here (we infer support from edge_index)
          - 'features'      data.x
          - 'labels'        data.y
          - 'labels_mask'   e.g. data.train_mask
          - 'dropout'       dropout probability (float)
          - 'weight_decay'  L2 coefficient (float)
        input_dim:  feature dimension F
        hidden_dim: hidden units H
        """
        super(GCN, self).__init__(name=kwargs.get('name'),
                                  logging=kwargs.get('logging', False))
        self.placeholders = placeholders
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = placeholders['labels'].shape[1]
        self.dropout    = placeholders.get('dropout', 0.0)
        self.weight_decay = placeholders.get('weight_decay', 0.0)

    def _build(self):
        # Layer 1: GCNConv(F→H) + ReLU + Dropout
        conv1 = GCNConv(self.input_dim, self.hidden_dim, cached=True)
        self.layers.append(conv1)
        # Layer 2: GCNConv(H→C) linear
        conv2 = GCNConv(self.hidden_dim, self.output_dim, cached=True)
        self.layers.append(conv2)

    def loss(self, logits, labels, mask):
        """
        logits: [N, C], labels: [N, C] one-hot or [N] int?
        mask: boolean mask [N]
        """
        # If one-hot labels, convert to class indices
        if labels.dim() == 2:
            labels_idx = labels.argmax(dim=1)
        else:
            labels_idx = labels
        # masked cross‐entropy
        loss = F.cross_entropy(logits[mask], labels_idx[mask])
        # weight decay on first layer
        wd = self.layers[0].lin.weight.norm(2) ** 2
        return loss + self.weight_decay * wd

    def accuracy(self, logits, labels, mask):
        # Override to accept one-hot or indices
        if labels.dim() == 2:
            labels_idx = labels.argmax(dim=1)
        else:
            labels_idx = labels
        pred = logits[mask].argmax(dim=1)
        return (pred == labels_idx[mask]).float().mean()
