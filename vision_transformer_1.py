from typing import Optional, Tuple
import torch
import torch.nn as nn

# This file implements a Vision Transformer (ViT) model that processes images by:
# 1. Splitting the image into patches
# 2. Converting each patch into an embedding
# 3. Adding positional embeddings
# 4. Processing through transformer layers
# 5. Outputting final image features

class VisionConfig:
    """Configuration class that stores all the hyperparameters needed for the vision model"""

    def __init__(
        self,
        hidden_size=768,  # Size of the embeddings used throughout the model
        intermediate_size=3072,  # Size of the intermediate layer in MLP
        num_hidden_layers=12,  # Number of transformer layers
        num_attention_heads=12,  # Number of attention heads in each transformer layer
        num_channels=3,  # Number of input image channels (3 for RGB)
        image_size=224,  # Input image size (224x224 pixels)
        patch_size=16,  # Size of each image patch (16x16 pixels)
        layer_norm_eps=1e-6,  # Small constant for numerical stability in layer norm
        attention_dropout=0.0,  # Dropout rate for attention
        num_image_tokens: int = None,  # Number of image tokens (patches) - calculated as (image_size/patch_size)^2
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class VisionEmbeddings(nn.Module):
    """Converts input images into patch embeddings and adds positional embeddings"""
    
    def __init__(self, config: VisionConfig):
        """Initialize the vision embeddings module.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Conv2d layer that splits image into patches and projects them to embedding dimension
        # For example: 224x224 image with 16x16 patches -> 14x14=196 patches
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        # Calculate total number of patches (e.g., 196 for 224x224 image with 16x16 patches)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # Create learnable position embeddings for each patch
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # Create fixed position IDs tensor [0, 1, 2, ..., num_patches-1]
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Convert input images into patch embeddings with positional encoding.
        
        Args:
            pixel_values (torch.FloatTensor): Input images of shape [Batch_Size, Channels, Height, Width]
            
        Returns:
            torch.Tensor: Patch embeddings with positional encoding of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        # This helps the model understand the relative position of patches
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class Attention(nn.Module):
    """Multi-headed attention mechanism that allows the model to focus on different parts of the input"""

    def __init__(self, config):
        """Initialize multi-headed attention module.
        
        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Split embedding dimension among heads (e.g., 768/12 = 64)
        self.head_dim = self.embed_dim // self.num_heads
        # Scaling factor for dot product attention
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        # Linear projections for Query, Key, Value
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # Final output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply multi-headed self-attention to the input.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim]
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Attention output of shape [Batch_Size, Num_Patches, Embed_Dim]
                - Attention weights of shape [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        """
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # Project input into Query, Key, Value vectors
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V to separate the heads
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores using scaled dot-product attention
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply softmax to get attention probabilities
        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply attention weights with values to get the final attention output
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Reshape output back to original dimensions
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # Final projection to combine all heads
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    """Multi-Layer Perceptron that processes each patch independently"""
    
    def __init__(self, config):
        """Initialize MLP module.
        
        Args:
            config: Configuration object containing MLP parameters
        """
        super().__init__()
        self.config = config
        # First fully connected layer expands dimension
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second fully connected layer projects back to original dimension
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to input features.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim]
            
        Returns:
            torch.Tensor: Transformed features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Expand dimension and apply GELU activation
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # Project back to original dimension
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class EncoderLayer(nn.Module):
    """Single transformer layer combining attention and MLP with residual connections"""
    
    def __init__(self, config: VisionConfig):
        """Initialize a single transformer encoder layer.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        # Multi-head self attention layer
        self.self_attn = Attention(config)
        # Layer normalization before attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # MLP block
        self.mlp = MLP(config)
        # Layer normalization before MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Process input through self-attention and MLP with residual connections.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim]
            
        Returns:
            torch.Tensor: Processed features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # First residual block: Self-attention
        # Save input for residual connection
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # Layer norm before attention
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # Apply self-attention
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # Add residual connection
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        # Second residual block: MLP
        # Save input for residual connection
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # Layer norm before MLP
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # Apply MLP
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # Add residual connection
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class Encoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, config: VisionConfig):
        """Initialize the transformer encoder with multiple layers.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        # Create list of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Process input through all encoder layers sequentially.
        
        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape [Batch_Size, Num_Patches, Embed_Dim]
            
        Returns:
            torch.Tensor: Encoded features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Process input through each encoder layer sequentially
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class VisionTransformer(nn.Module):
    """Main vision transformer model combining embeddings, encoder, and final layer norm"""
    
    def __init__(self, config: VisionConfig):
        """Initialize the complete vision transformer model.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Create embedding layer for converting image patches to embeddings
        self.embeddings = VisionEmbeddings(config)
        # Create encoder with multiple transformer layers
        self.encoder = Encoder(config)
        # Final layer normalization
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process images through the complete vision transformer pipeline.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape [Batch_Size, Channels, Height, Width]
            
        Returns:
            torch.Tensor: Final encoded features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Convert image to patch embeddings
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        # Process through transformer encoder
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Final layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class VisionModel(nn.Module):
    """Wrapper class for the vision transformer model"""

    def __init__(self, config: VisionConfig):
        """Initialize the vision model wrapper.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """Process images through the vision transformer model.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape [Batch_Size, Channels, Height, Width]
            
        Returns:
            torch.Tensor: Encoded image features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Process image through vision transformer
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 
