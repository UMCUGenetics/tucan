import torch.nn as nn

ACT_FUNC = {
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

"""Simple fully forward model
"""
from typing import Dict

import torch
from torch import nn

#from sturgeoff.models.abstract import AbstractModel
#from sturgeoff.models.nn import SignEmbedding, ACT_FUNC

class SignEmbedding(nn.Module):
    """Embedding layer that embeds 0, 1, 2 into 0, 1, -1 respectively
    """
    def __init__(self, *args, **kwargs):
        super(SignEmbedding, self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(3, 1, 0, *args, **kwargs)
        self.embedding.weight = nn.Parameter(
            torch.tensor([[0.], [1.], [-1.]]), 
            requires_grad=False
        )
    def forward(self, x):
        return x
        #return self.embedding(x)


class SturgeonSubmodel(nn.Module):
    """Simple sequential model 
    
    Forwards through a bunch of linear layers with activation functions in 
    between. Does not have an activation after the last linear layer.

    Args:
        in_size: input size of the first layer
        activation: activation function to use
        classification_sizes: a dict indicating the size of each level in 
        decoding_dict: dict that maps integers to classes
        bias: whether to use bias in the linear layers
    """

    def __init__(
        self, 
        in_size: int,
        activation: str,
        classification_sizes: Dict[str, int],
        #decoding_dict: Dict[int, str],
        bias: bool = True,
        *args, 
        **kwargs,
    ):
        super(SturgeonSubmodel, self).__init__(*args, **kwargs)

        self.embedding_layers = nn.Sequential(
            SignEmbedding(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_size, 256, bias=bias),
            nn.Dropout(0.5),
            ACT_FUNC[activation],
            nn.Linear(256, 128, bias=bias),
            nn.Dropout(0.5),
            ACT_FUNC[activation],
        )
        self.classification_layers = torch.nn.ModuleDict()
        self.classification_layers['p:ce:type'] = nn.Linear(128, classification_sizes)
        
        #self.decoding_dict = decoding_dict

        self.init_parameters = {
            'in_size': in_size,
            'activation': activation,
            #'decoding_dict': decoding_dict,
            'classification_sizes': classification_sizes,
            'bias': bias,
        }

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward through the layers

        Args:
            x: input tensor of shape [B, H]

        Returns a tensor of shape [B, out_size]
        """

        output = dict()
        emb = self.embedding_layers(x)

        #output['emb:ce:type'] = emb
        
        output['y'] = self.classification_layers['p:ce:type'](emb)
    
        return output
      
