import torch
import torch.nn as nn
from math import sqrt
import math
from .parametrizations import convert_array
from .circuits import *
from torch.utils.data import Dataset
import warnings

device = 'cuda'  # Set the device for computations

#################### 1st Hybrid Approach ####################
class EncoderLayer_hybrid1(nn.Module):
    """
    Encoder layer using a hybrid attention mechanism with quantum circuits.
    
    Args:
        Token_Dim (int): Token embedding dimension.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of attention heads.
        ff_dim (int, optional): Feedforward dimension.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension, ff_dim=None):
        super(EncoderLayer_hybrid1, self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim // head_dimension
        
        # Define attention heads with quantum-enhanced AttentionHead_Hybrid2
        self.heads = nn.ModuleList([AttentionHead_Hybrid2(Token_Dim, self.MultiHead_Embed_Dim) for i in range(head_dimension)])
        
        # Use a quantum-based merger layer (QLayer) instead of a classical feedforward network
        self.merger = QLayer(measure_value, [3 * Embed_Dim], int(Embed_Dim))
        
        # Apply layer normalization
        self.norm1 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)

    def forward(self, input1):
        """
        Forward pass with quantum-enhanced multi-head attention and residual connection.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output after applying attention and merger layers.
        """
        input1_norm = self.norm1(input1)  # Normalize input
        # Concatenate output of each attention head along the last dimension
        head_result = torch.cat([m(input1_norm[..., (i * self.MultiHead_Embed_Dim):((i + 1) * self.MultiHead_Embed_Dim)]) for i, m in enumerate(self.heads)], dim=-1)
        # Flatten and apply merger layer, then reshape to match input and add residual
        res = self.merger(head_result.flatten(0, 1)).reshape(head_result.shape) + input1
        return res


#################### 2nd Hybrid Approach ####################
class EncoderLayer_hybrid2(nn.Module):
    """
    Encoder layer using an alternative hybrid approach with classical feedforward layer.
    
    Args:
        Token_Dim (int): Token embedding dimension.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of attention heads.
        ff_dim (int, optional): Feedforward dimension.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension, ff_dim=None):
        super(EncoderLayer_hybrid2, self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim // head_dimension
        
        # Define attention heads with quantum-enhanced AttentionHead_Hybrid2
        self.heads = nn.ModuleList(
            [AttentionHead_Hybrid2(Token_Dim, self.MultiHead_Embed_Dim) for _ in range(head_dimension)]
        )
        
        # Use a classical feedforward network for the merger layer
        self.merger = construct_FNN(
            input_size=Embed_Dim, 
            layers=[ff_dim, Embed_Dim], 
            activation=nn.GELU
        )
        self.norm1 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)

    def forward(self, input1):
        input1_norm = self.norm1(input1)
        head_result = torch.cat(
            [
                m(input1_norm[..., (i * self.MultiHead_Embed_Dim):((i + 1) * self.MultiHead_Embed_Dim)]) 
                for i, m in enumerate(self.heads)
            ], 
            dim=-1
        )
        res = self.merger(head_result) + input1  # Apply merger and add residual
        return res


#################### Attention Mechanisms ####################
class AttentionHead_Hybrid2(nn.Module):
    """
    Quantum-enhanced attention head with query, key, and value computed through quantum layers.

    Args:
        Token_Dim (int): Token embedding dimension.
        MultiHead_Embed_Dim (int): Multi-head embedding dimension.
    """
    def __init__(self, Token_Dim, MultiHead_Embed_Dim):
        super(AttentionHead_Hybrid2, self).__init__()
        self.MultiHead_Embed_Dim = MultiHead_Embed_Dim
        
        self.norm = nn.LayerNorm(MultiHead_Embed_Dim, elementwise_affine=False)
        
        # Define quantum layers for query, key, and value
        self.V = QLayer(measure_value, [3 * MultiHead_Embed_Dim], int(MultiHead_Embed_Dim))
        self.Q = QLayer(measure_query_key, [3 * MultiHead_Embed_Dim + 1], int(MultiHead_Embed_Dim))
        self.K = QLayer(measure_query_key, [3 * MultiHead_Embed_Dim + 1], int(MultiHead_Embed_Dim))
        
        # Attention mechanism using scaled softmax
        self.attention = lambda A, V: torch.bmm(nn.Softmax(dim=-1)(A / MultiHead_Embed_Dim ** 0.5), V)
        self.flattener = lambda A: A.flatten(0, 1)

    def forward(self, input1):
        """
        Forward pass to compute attention weights and apply them to values.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Attention output.
        """
        flat_input = self.flattener(input1)  # Flatten input for quantum processing

        # Compute value, query, and key using quantum layers
        V = self.V(flat_input).reshape(input1.shape)
        Q = self.Q(flat_input).reshape(*input1.shape[:2])
        K = self.K(flat_input).reshape(*input1.shape[:2])

        # Compute attention weights as squared differences between Q and K
        A = -(Q.unsqueeze(-2) - K.unsqueeze(-3)) ** 2
        return self.attention(A, V)


#################### Classical Approach ####################
class EncoderLayer(nn.Module):
    """
    Classical encoder layer using multi-head attention and feedforward network.

    Args:
        Token_Dim (int): Token embedding dimension.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of attention heads.
        ff_dim (int): Feedforward network dimension.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension, ff_dim):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)
        self.norm2 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)
        
        # Define multi-head attention layer
        self.MHA = MultiHead(Token_Dim, Embed_Dim, head_dimension)
        
        # Define classical feedforward merger layer
        self.merger = construct_FNN([ff_dim, Embed_Dim], activation=nn.GELU)

    def forward(self, input1):
        """
        Forward pass with multi-head attention and residual connection.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output after attention and merger layers.
        """
        input1_norm = self.norm1(input1)  # Normalize input
        res = self.MHA(input1_norm) + input1  # Apply attention and add residual
        return self.merger(self.norm2(res)) + res


class MultiHead(nn.Module):
    """
    Classical multi-head attention layer.

    Args:
        Token_Dim (int): Token embedding dimension.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of heads.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension):
        super(MultiHead, self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim // head_dimension
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([AttentionHead(Token_Dim, self.MultiHead_Embed_Dim) for i in range(head_dimension)])

    def forward(self, input1):
        """
        Concatenates the output of each attention head.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Concatenated outputs from all heads.
        """
        return torch.cat([m(input1[..., (i * self.MultiHead_Embed_Dim):((i + 1) * self.MultiHead_Embed_Dim)]) for i, m in enumerate(self.heads)], dim=-1)


class AttentionHead(nn.Module):
    """
    Classical attention head using linear layers for query, key, and value.

    Args:
        Token_Dim (int): Token embedding dimension.
        embed_per_head_dim (int): Dimension of each attention head.
    """
    def __init__(self, Token_Dim, embed_per_head_dim):
        super(AttentionHead, self).__init__()
        # Linear layers for query, key, and value
        self.Q = nn.Linear(embed_per_head_dim, embed_per_head_dim, bias=False)
        self.V = nn.Linear(embed_per_head_dim, embed_per_head_dim, bias=False)
        self.K = nn.Linear(embed_per_head_dim, embed_per_head_dim, bias=False)
        self.soft = nn.Softmax(dim=-1)

    def attention(self, Q, K, V):
        return torch.bmm(self.soft(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Q.shape[-1])), V)

    def forward(self, input1):
        """
        Computes attention weights and applies them to values.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output of attention head.
        """
        Q = self.Q(input1)
        K = self.K(input1)
        V = self.V(input1)
        return self.attention(Q, K, V)


############################### Transformer Architecture ###############################
class Transformer(nn.Module):
    """
    Transformer model with various encoder layers for hybrid and classical attention.

    Args:
        Token_Dim (int): Token embedding dimension.
        Image_Dim (int): Image embedding dimension.
        head_dimension (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        Embed_Dim (int): Embedding dimension.
        ff_dim (int): Feedforward network dimension.
        pos_embedding (bool): Use positional embedding if True.
        classifying_type (str): Type of classification (e.g., 'cls_token').
        attention_type (str): Type of attention (e.g., 'hybrid1').
    """
    def __init__(self, Token_Dim, Image_Dim, head_dimension, n_layers, Embed_Dim, ff_dim, pos_embedding, classifying_type, attention_type):
        super(Transformer, self).__init__()
        self.cls_type = classifying_type
        self.embedding = pos_embedding
        
        # Initialize positional embeddings
        self.pos_embedding = nn.parameter.Parameter(torch.tensor([math.sin(1 / 10000 ** ((i - 1) / Embed_Dim)) if i % 2 == 1 else math.cos(i / 10000 ** ((i - 1) / Embed_Dim)) for i in range(Embed_Dim)]))
        self.pos_embedding.requires_grad = False

        # Map attention type to corresponding encoder layer
        attention_dict = {'hybrid2': EncoderLayer_hybrid2, 'classic': EncoderLayer, 'hybrid1': EncoderLayer_hybrid1}
        if self.cls_type == 'cls_token':
            Token_Dim += 1  # Add a class token if required
        
        # Initialize encoder layers
        self.encoder_layers = nn.ModuleList([attention_dict[attention_type](Token_Dim, Embed_Dim, head_dimension, ff_dim) for i in range(n_layers)])
        
        # Initialize class token for 'cls_token' classification type
        if self.cls_type == "cls_token":
            self.class_token = nn.parameter.Parameter(torch.rand(Embed_Dim, dtype=torch.float32).abs().to('cuda') / math.sqrt(Embed_Dim))
        
        # Embedding layer for input images
        self.embedder = nn.Linear(Image_Dim, Embed_Dim)
        
        # Define final activation method based on classification type
        if self.cls_type == 'max':
            self.final_act = lambda temp: temp[-1].max(axis=1).values
        if self.cls_type == 'mean':
            self.final_act = lambda temp: temp[-1].mean(axis=1)
        if self.cls_type == 'sum':
            self.final_act = lambda temp: temp[-1].sum(axis=1)
        if self.cls_type == 'cls_token':
            self.final_act = lambda temp: temp[-1][:, 0]

    def forward(self, input1):
        """
        Forward pass through the transformer.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Final output of the transformer.
        """
        if self.cls_type == "cls_token":
            cls_token = self.class_token.expand(input1.shape[0], 1, -1)
            input1_ = torch.cat((cls_token, self.embedder(input1)), axis=1)
        else:
            input1_ = self.embedder(input1)
        
        # Add positional embedding if enabled
        if self.embedding:
            temp = [input1_ + self.pos_embedding[None, None, :]]
        else:
            temp = [input1_]
        
        # Pass through encoder layers
        for i, m in enumerate(self.encoder_layers):
            temp.append(m(temp[i]))

        return self.final_act(temp)


class HViT(nn.Module):
    """
    Hybrid Vision Transformer (HViT) model combining a transformer and a classifier.

    Args:
        Token_Dim (int): Token embedding dimension.
        Image_Dim (int): Image embedding dimension.
        head_dimension (int): Number of attention heads.
        n_layers (int): Number of transformer layers.
        FC_layers (list): List of dimensions for fully connected layers.
        attention_type (str): Type of attention.
        pos_embedding (bool): Use positional embedding if True.
        classifying_type (str): Type of classification (e.g., 'cls_token').
        Embed_Dim (int): Embedding dimension.
        ff_dim (int): Feedforward network dimension.
    """
    class HViT(nn.Module):
        def __init__(self, Token_Dim, Image_Dim, head_dimension, n_layers, FC_layers, attention_type, pos_embedding, classifying_type, Embed_Dim, ff_dim):
            super(HViT, self).__init__()
            self.transformer = Transformer(
                Token_Dim, Image_Dim, head_dimension, n_layers, 
                Embed_Dim, ff_dim, pos_embedding, classifying_type, attention_type
            )
            # Specify the input_size explicitly
            self.classifier = construct_FNN(
                input_size=Embed_Dim,
                layers=FC_layers,
                activation=nn.LeakyReLU
            )
    
    
        def forward(self, input1):
            """
            Forward pass through the HViT model.
    
            Args:
                input1 (Tensor): Input tensor.
    
            Returns:
                Tensor: Classification output.
            """
            return self.classifier(self.transformer(input1))

def construct_FNN(input_size, layers, activation=nn.GELU, output_activation=None, Dropout=None):
    """
    Constructs a fully connected neural network (FNN) with specified activations and dropout.

    Args:
        input_size (int): Size of the input features.
        layers (list): List specifying the size of each layer.
        activation (nn.Module): Activation function to use.
        output_activation (nn.Module, optional): Output activation function.
        Dropout (float, optional): Dropout probability.

    Returns:
        nn.Sequential: Fully connected neural network.
    """
    layer_list = []
    last_size = input_size
    for size in layers:
        layer_list.append(nn.Linear(last_size, size))
        layer_list.append(activation())
        last_size = size
    if Dropout:
        layer_list.insert(len(layer_list) - 2, nn.Dropout(Dropout))
    if output_activation is not None:
        layer_list.append(output_activation)
    # Remove the last activation if not desired
    return nn.Sequential(*layer_list[:-1])


'''
Old construct_FNN
def construct_FNN(layers, activation=nn.GELU, output_activation=None, Dropout=None):
    """
    Constructs a fully connected neural network (FNN) with specified activations and dropout.

    Args:
        layers (list): List specifying the size of each layer.
        activation (nn.Module): Activation function to use.
        output_activation (nn.Module, optional): Output activation function.
        Dropout (float, optional): Dropout probability.

    Returns:
        nn.Sequential: Fully connected neural network.
    """
    layer = [j for i in layers for j in [nn.LazyLinear(i), activation()]][:-1]
    if Dropout:
        layer.insert(len(layer) - 2, nn.Dropout(Dropout))
    if output_activation is not None:
        layer.append(output_activation)
    return nn.Sequential(*layer)
'''
