import torch
import torch.nn as nn
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
        self.heads = nn.ModuleList([AttentionHead_Hybrid2(Token_Dim, self.MultiHead_Embed_Dim) for _ in range(head_dimension)])
        
        # Use a quantum-based merger layer (QLayer) instead of a classical feedforward network
        self.merger = QLayer(measure_value, [3 * Embed_Dim], int(Embed_Dim))
        
        # Apply layer normalization
        self.norm1 = nn.LayerNorm(Embed_Dim, elementwise_affine=False)

    def forward(self, input1):
        """
        Forward pass with quantum-enhanced multi-head attention and residual connection.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output after applying attention and merger layers.
        """
        input1_norm = self.norm1(input1)  # Normalize input

        # Split the input for each head
        head_inputs = torch.chunk(input1_norm, len(self.heads), dim=-1)

        # Process all heads and concatenate results
        head_outputs = torch.cat([head(head_input) for head, head_input in zip(self.heads, head_inputs)], dim=-1)

        # Flatten and apply merger layer, then reshape to match input and add residual
        batch_size, seq_len, embed_dim = head_outputs.shape
        merger_input = head_outputs.reshape(batch_size * seq_len, -1)
        merged = self.merger(merger_input).reshape(batch_size, seq_len, -1)
        res = merged + input1
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
        self.heads = nn.ModuleList([AttentionHead_Hybrid2(Token_Dim, self.MultiHead_Embed_Dim) for _ in range(head_dimension)])
        
        # Use a classical feedforward network for the merger layer
        self.merger = construct_FNN([ff_dim, Embed_Dim], activation=nn.GELU)
        self.norm1 = nn.LayerNorm(Embed_Dim, elementwise_affine=False)

    def forward(self, input1):
        """
        Forward pass with quantum-enhanced multi-head attention and classical feedforward network.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output after applying attention and merger layers.
        """
        input1_norm = self.norm1(input1)  # Normalize input

        # Split the input for each head
        head_inputs = torch.chunk(input1_norm, len(self.heads), dim=-1)

        # Process all heads and concatenate results
        head_outputs = torch.cat([head(head_input) for head, head_input in zip(self.heads, head_inputs)], dim=-1)

        res = self.merger(head_outputs) + input1  # Apply merger and add residual
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

    def forward(self, input1):
        """
        Forward pass to compute attention weights and apply them to values.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Attention output.
        """
        batch_size, seq_len, embed_dim = input1.shape
        input1_flat = input1.reshape(batch_size * seq_len, embed_dim)  # Flatten input for quantum processing

        # Compute value, query, and key using quantum layers
        V = self.V(input1_flat).reshape(batch_size, seq_len, -1)
        Q = self.Q(input1_flat).reshape(batch_size, seq_len)
        K = self.K(input1_flat).reshape(batch_size, seq_len)

        # Vectorized computation of attention weights
        Q_expanded = Q.unsqueeze(2)  # Shape: [batch_size, seq_len, 1]
        K_expanded = K.unsqueeze(1)  # Shape: [batch_size, 1, seq_len]
        A = -(Q_expanded - K_expanded) ** 2  # Shape: [batch_size, seq_len, seq_len]

        # Compute attention output
        output = self.attention(A, V)
        return output


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
        self.norm1 = nn.LayerNorm(Embed_Dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(Embed_Dim, elementwise_affine=False)
        
        # Define multi-head attention layer
        self.MHA = MultiHead(Token_Dim, Embed_Dim, head_dimension)
        
        # Define classical feedforward merger layer
        self.merger = construct_FNN([Embed_Dim, ff_dim, Embed_Dim], activation=nn.GELU)

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
        self.head_dimension = head_dimension
        self.embed_per_head_dim = Embed_Dim // head_dimension
        
        # Combined linear layers for all heads
        self.Q = nn.Linear(Embed_Dim, Embed_Dim, bias=False)
        self.K = nn.Linear(Embed_Dim, Embed_Dim, bias=False)
        self.V = nn.Linear(Embed_Dim, Embed_Dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input1):
        """
        Computes attention weights and applies them to values.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Output of attention heads concatenated.
        """
        batch_size, seq_len, embed_dim = input1.size()
        Q = self.Q(input1).view(batch_size, seq_len, self.head_dimension, self.embed_per_head_dim).transpose(1, 2)
        K = self.K(input1).view(batch_size, seq_len, self.head_dimension, self.embed_per_head_dim).transpose(1, 2)
        V = self.V(input1).view(batch_size, seq_len, self.head_dimension, self.embed_per_head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_per_head_dim)
        weights = self.softmax(scores)
        context = torch.matmul(weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return context


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
        self.pos_embedding = nn.Parameter(torch.zeros(1, Token_Dim, Embed_Dim))
        nn.init.uniform_(self.pos_embedding, -0.1, 0.1)
        self.pos_embedding.requires_grad = True

        # Map attention type to corresponding encoder layer
        attention_dict = {'hybrid2': EncoderLayer_hybrid2, 'classic': EncoderLayer, 'hybrid1': EncoderLayer_hybrid1}
        if self.cls_type == 'cls_token':
            Token_Dim += 1  # Add a class token if required
        
        # Initialize encoder layers
        self.encoder_layers = nn.ModuleList([attention_dict[attention_type](Token_Dim, Embed_Dim, head_dimension, ff_dim) for _ in range(n_layers)])
        
        # Initialize class token for 'cls_token' classification type
        if self.cls_type == "cls_token":
            self.class_token = nn.Parameter(torch.zeros(1, 1, Embed_Dim))
            nn.init.uniform_(self.class_token, -0.1, 0.1)
        
        # Embedding layer for input images
        self.embedder = nn.Linear(Image_Dim, Embed_Dim)
        
        # Define final activation method based on classification type
        if self.cls_type == 'max':
            self.final_act = lambda temp: temp.max(dim=1).values
        elif self.cls_type == 'mean':
            self.final_act = lambda temp: temp.mean(dim=1)
        elif self.cls_type == 'sum':
            self.final_act = lambda temp: temp.sum(dim=1)
        elif self.cls_type == 'cls_token':
            self.final_act = lambda temp: temp[:, 0]

    def forward(self, input1):
        """
        Forward pass through the transformer.

        Args:
            input1 (Tensor): Input tensor.

        Returns:
            Tensor: Final output of the transformer.
        """
        batch_size = input1.size(0)
        input_embedded = self.embedder(input1)
        
        if self.cls_type == "cls_token":
            cls_tokens = self.class_token.expand(batch_size, -1, -1)
            input1_ = torch.cat((cls_tokens, input_embedded), dim=1)
        else:
            input1_ = input_embedded
        
        # Add positional embedding if enabled
        if self.embedding:
            input1_ = input1_ + self.pos_embedding[:, :input1_.size(1), :]

        x = input1_
        # Pass through encoder layers
        for m in self.encoder_layers:
            x = m(x)

        return self.final_act(x)
    
    
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
        x = self.transformer(input1)
        return self.classifier(x)
    
    
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
        layer.append(output_activation())
    return nn.Sequential(*layer)
