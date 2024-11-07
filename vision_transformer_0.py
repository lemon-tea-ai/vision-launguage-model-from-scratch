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
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass


class Attention(nn.Module):
    """Multi-headed attention mechanism that allows the model to focus on different parts of the input"""

    def __init__(self, config):
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
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass


class MLP(nn.Module):
    """Multi-Layer Perceptron that processes each patch independently"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # First fully connected layer expands dimension
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second fully connected layer projects back to original dimension
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass


class EncoderLayer(nn.Module):
    """Single transformer layer combining attention and MLP with residual connections"""
    
    def __init__(self, config: VisionConfig):
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
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass


class Encoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, config: VisionConfig):
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
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass


class VisionTransformer(nn.Module):
    """Main vision transformer model combining embeddings, encoder, and final layer norm"""
    
    def __init__(self, config: VisionConfig):
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
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass


class VisionModel(nn.Module):
    """Wrapper class for the vision transformer model"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        pass
