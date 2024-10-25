import torch
import torch.nn as nn
from math import sqrt
import math
from .parametrizations import convert_array
from .circuits import *
from torch.utils.data import Dataset
import warnings

#################### 1st Hybrid Approach
class AttentionHead_Hybrid1(nn.Module):
    """
    Implements a single attention head using the first hybrid approach that combines
    classical and quantum methods.

    Args:
        Token_Dim (int): Dimension of token embeddings.
        MultiHead_Embed_Dim (int): Dimension of embeddings per head.

    Attributes:
        V (nn.Linear): Linear layer for value transformation.
        norm (nn.LayerNorm): Layer normalization.
        A (callable): Quantum attention circuit.
        attention (callable): Attention function that combines softmax normalization with values.
    """
    def __init__(self, Token_Dim, MultiHead_Embed_Dim):
        super(AttentionHead_Hybrid1, self).__init__()
        self.V = nn.Linear(MultiHead_Embed_Dim, MultiHead_Embed_Dim)
        self.norm = nn.LayerNorm(MultiHead_Embed_Dim, elementwise_affine=False)

        len_phi = 2 * MultiHead_Embed_Dim - 3  # Parameter size for quantum attention circuit
        self.A = circuit_to_layer(compute_attention_element, list(range(MultiHead_Embed_Dim)), {'phi': len_phi})
        self.attention = lambda V, A: torch.bmm(nn.Softmax(dim=-1)(A / math.sqrt(MultiHead_Embed_Dim)), V)

    def forward(self, input1):
        """
        Forward pass that computes value and attention scores using the quantum attention circuit.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: The output of attention head after normalization.
        """
        input2 = convert_array(input1)  # Quantum-compatible data conversion
        input3 = (input1 ** 2).sum(axis=-1).sqrt()  # Compute norms for normalization

        V = self.V(input1)
        A = compute_attention(input2, input3, self.A)

        return self.norm(self.attention(V, A) + input1)


class MultiHead_hybrid1(nn.Module):
    """
    Implements multi-head attention by stacking multiple AttentionHead_Hybrid1 heads.

    Args:
        Token_Dim (int): Dimension of token embeddings.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of attention heads.
        attention_type (str): Type of attention used.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension, attention_type):
        super(MultiHead_hybrid1, self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim // head_dimension
        self.heads = nn.ModuleList([AttentionHead_Hybrid1(Token_Dim, self.MultiHead_Embed_Dim) for i in range(head_dimension)])

    def forward(self, input1):
        """
        Concatenates the output of each attention head.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Concatenated outputs from all heads.
        """
        return torch.cat([m(input1[..., (i * self.MultiHead_Embed_Dim):((i + 1) * self.MultiHead_Embed_Dim)]) for i, m in enumerate(self.heads)], dim=-1)


class EncoderLayer_hybrid1(nn.Module):
    """
    Encoder layer using the first hybrid approach with multi-head attention and feedforward network.

    Args:
        Token_Dim (int): Token embedding dimension.
        Image_Dim (int): Image embedding dimension.
        head_dimension (int): Number of attention heads.
        attention_type (str): Type of attention mechanism.
        ff_dim (int, optional): Feedforward dimension.
    """
    def __init__(self, Token_Dim, Image_Dim, head_dimension, attention_type, ff_dim=None):
        super(EncoderLayer_hybrid1, self).__init__()
        self.norm1 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)
        self.norm2 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)
        self.MHA = MultiHead_hybrid1(Token_Dim, Embed_Dim, head_dimension, attention_type)
        self.merger = construct_FNN([Embed_Dim, Embed_Dim], activation=nn.GELU)

    def forward(self, input1):
        """
        Forward pass with residual connection and feedforward network.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output of the encoder layer.
        """
        input1_norm = self.norm1(input1)
        res = self.MHA(input1_norm) + input1
        return self.merger(self.norm2(res)) + res


#################### 2nd Hybrid Approach ######################
class EncoderLayer_hybrid2(nn.Module):
    """
    Encoder layer using the second hybrid approach with multi-head attention and feedforward network.

    Args:
        Token_Dim (int): Token embedding dimension.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of attention heads.
        ff_dim (int, optional): Feedforward dimension, unused here.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension, ff_dim=None):
        if ff_dim is not None:
            warnings.warn("ff_dim is not utilized since no ff")
        super(EncoderLayer_hybrid2, self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim // head_dimension
        self.heads = nn.ModuleList([AttentionHead_Hybrid2(Token_Dim, self.MultiHead_Embed_Dim) for i in range(head_dimension)])
        self.merger = construct_FNN([head_dimension, head_dimension])
        self.norm1 = nn.LayerNorm([Embed_Dim], elementwise_affine=False)

    def forward(self, input1):
        """
        Applies multi-head attention and feedforward network with a residual connection.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output of the encoder layer.
        """
        input1_norm = self.norm1(input1)
        head_result = torch.stack([m(input1_norm[..., (i * self.MultiHead_Embed_Dim):((i + 1) * self.MultiHead_Embed_Dim)]) for i, m in enumerate(self.heads)], dim=-1)
        res = self.merger(head_result).flatten(start_dim=-2) + input1
        return res


class AttentionHead_Hybrid2(nn.Module):
    """
    Attention head in the second hybrid approach using quantum circuits for value, query, and key computations.

    Args:
        Token_Dim (int): Token embedding dimension.
        MultiHead_Embed_Dim (int): Multi-head embedding dimension.
    """
    def __init__(self, Token_Dim, MultiHead_Embed_Dim):
        super(AttentionHead_Hybrid2, self).__init__()
        self.MultiHead_Embed_Dim = MultiHead_Embed_Dim
        self.norm = nn.LayerNorm(MultiHead_Embed_Dim)
        self.V = QLayer(measure_value, [3 * MultiHead_Embed_Dim], int(MultiHead_Embed_Dim))
        self.Q = QLayer(measure_query_key, [3 * MultiHead_Embed_Dim + 1], int(MultiHead_Embed_Dim))
        self.K = QLayer(measure_query_key, [3 * MultiHead_Embed_Dim + 1], int(MultiHead_Embed_Dim))
        self.attention = lambda A, V: torch.bmm(nn.Softmax(dim=-1)(A / MultiHead_Embed_Dim ** 0.5), V)
        self.flattener = lambda A: A.flatten(0, 1)

    def forward(self, input1):
        """
        Computes attention with value, query, and key obtained from quantum circuits.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Attention output.
        """
        flat_input = self.flattener(input1)
        V = self.V(flat_input).reshape(input1.shape).type(input1.dtype)
        Q = self.Q(flat_input).reshape(*input1.shape[:2])
        K = self.K(flat_input).reshape(*input1.shape[:2])

        # Compute attention weights A based on Q and K differences
        A = torch.empty((*input1.shape[:-1], input1.shape[-2]), device=input1.device)
        for j in range(input1.shape[-2]):
            A[..., j] = -(Q - K[..., j][..., None]) ** 2
        return self.attention(A, V)


#################### Classical Approach ######################
class AttentionHead(nn.Module):
    """
    Classical attention head using linear layers for query, key, and value.

    Args:
        Token_Dim (int): Token embedding dimension.
        embed_per_head_dim (int): Embedding dimension per head.
    """
    def __init__(self, Token_Dim, embed_per_head_dim):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(embed_per_head_dim, embed_per_head_dim)
        self.V = nn.Linear(embed_per_head_dim, embed_per_head_dim)
        self.K = nn.Linear(embed_per_head_dim, embed_per_head_dim)
        self.soft = nn.Softmax(dim=-1)

    def attention(self, Q, K, V):
        return torch.bmm(self.soft(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Q.shape[-1])), V)

    def forward(self, input1):
        Q = self.Q(input1)
        K = self.K(input1)
        V = self.V(input1)
        return self.attention(Q, K, V)


class MultiHead(nn.Module):
    """
    Multi-head attention implementation for the classical approach.

    Args:
        Token_Dim (int): Token embedding dimension.
        Embed_Dim (int): Embedding dimension.
        head_dimension (int): Number of heads.
    """
    def __init__(self, Token_Dim, Embed_Dim, head_dimension):
        super(MultiHead, self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim // head_dimension
        self.heads = nn.ModuleList([AttentionHead(Token_Dim, self.MultiHead_Embed_Dim) for i in range(head_dimension)])

    def forward(self, input1):
        return torch.cat([m(input1[..., (i * self.MultiHead_Embed_Dim):((i + 1) * self.MultiHead_Embed_Dim)]) for i, m in enumerate(self.heads)], dim=-1)


class EncoderLayer(nn.Module):
    """
    Encoder layer for the classical approach with multi-head attention and feedforward network.

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
        self.MHA = MultiHead(Token_Dim, Embed_Dim, head_dimension)
        self.merger = construct_FNN([ff_dim, Embed_Dim], activation=nn.GELU)

    def forward(self, input1):
        input1_norm = self.norm1(input1)
        res = self.MHA(input1_norm) + input1
        return self.merger(self.norm2(res)) + res


############################### Transformer Model ###############################
class Transformer(nn.Module):
    """
    Transformer architecture combining different encoder layers.

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
        
        self.pos_embedding = nn.parameter.Parameter(torch.tensor([math.sin(1 / 10000 ** ((i - 1) / Embed_Dim)) if i % 2 == 1 else math.cos(i / 10000 ** ((i - 1) / Embed_Dim)) for i in range(Embed_Dim)]))
        self.pos_embedding.requires_grad = False
        attention_dict = {'hybrid2': EncoderLayer_hybrid2, 'classic': EncoderLayer, 'hybrid1': EncoderLayer_hybrid1}
        
        if self.cls_type == 'cls_token':
            self.encoder_layers = nn.ModuleList([attention_dict[attention_type](Token_Dim + 1, Embed_Dim, head_dimension, ff_dim) for i in range(n_layers)])
        else:
            self.encoder_layers = nn.ModuleList([attention_dict[attention_type](Token_Dim, Embed_Dim, head_dimension, ff_dim) for i in range(n_layers)])
        
        if self.cls_type == "cls_token":
            self.class_token = nn.parameter.Parameter(torch.rand(Embed_Dim) / math.sqrt(Embed_Dim))
        
        self.embedder = nn.Linear(Image_Dim, Embed_Dim)
        
        # Define final activation type based on classifying type
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
        Forward pass through the transformer with optional positional embedding.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Final output of the transformer after classification.
        """
        if self.cls_type == "cls_token":
            input1_ = torch.cat((self.class_token.repeat(input1.shape[0], 1, 1), self.embedder(input1)), axis=1)
        else:
            input1_ = self.embedder(input1)
        
        temp = [input1_ + self.pos_embedding[None, None, :]] if self.embedding else [input1_]
        
        for i, m in enumerate(self.encoder_layers):
            temp.append(m(temp[i]))

        return self.final_act(temp)


class HViT(nn.Module):
    """
    HViT model that combines a Transformer with a fully connected classifier.

    Args:
        Token_Dim (int): Token embedding dimension.
        Image_Dim (int): Image embedding dimension.
        head_dimension (int): Number of attention heads.
        n_layers (int): Number of transformer layers.
        FC_layers (list): List of fully connected layer dimensions.
        attention_type (str): Type of attention.
        pos_embedding (bool): Use positional embedding if True.
        classifying_type (str): Type of classification (e.g., 'cls_token').
        Embed_Dim (int): Embedding dimension.
        ff_dim (int): Feedforward network dimension.
    """
    def __init__(self, Token_Dim, Image_Dim, head_dimension, n_layers, FC_layers, attention_type, pos_embedding, classifying_type, Embed_Dim, ff_dim):
        super(HViT, self).__init__()
        self.transformer = Transformer(Token_Dim, Image_Dim, head_dimension, n_layers, Embed_Dim, ff_dim, pos_embedding, classifying_type, attention_type)
        self.classifier = construct_FNN(FC_layers, activation=nn.LeakyReLU)

    def forward(self, input1):
        """
        Forward pass through the HViT model.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Classification output.
        """
        return self.classifier(self.transformer(input1))


def construct_FNN(layers, activation=nn.ReLU, output_activation=None, Dropout=None):
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
