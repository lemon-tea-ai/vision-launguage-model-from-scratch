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
        # TODO:
        # 1. Calculate num_patches = (image_size // patch_size) ** 2
        # 2. Create patch_embeddings as nn.Conv2d to convert image patches to embeddings
        #    - in_channels = num_channels
        #    - out_channels = hidden_size
        #    - kernel_size = patch_size
        #    - stride = patch_size
        # 3. Create position_embeddings as nn.Parameter of shape (1, num_patches, hidden_size)
        # 4. Create layer_norm as nn.LayerNorm with hidden_size and eps
        # 5. Create dropout layer with config.attention_dropout

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Convert input images into patch embeddings with positional encoding.
        
        Args:
            pixel_values (torch.FloatTensor): Input images of shape [Batch_Size, Channels, Height, Width]
            
        Returns:
            torch.Tensor: Patch embeddings with positional encoding of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # TODO:
        # 1. Apply patch_embeddings to convert image to patches
        # 2. Reshape output to (batch_size, num_patches, hidden_size)
        # 3. Add position_embeddings
        # 4. Apply layer_norm and dropout
        # 5. Return final embeddings


class Attention(nn.Module):
    """Multi-headed attention mechanism that allows the model to focus on different parts of the input"""

    def __init__(self, config):
        """Initialize multi-headed attention module.
        
        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        # TODO:
        # 1. Set head_size = hidden_size // num_attention_heads
        # 2. Create query, key, value projections as nn.Linear layers
        # 3. Create attention_dropout layer
        # 4. Create output projection as nn.Linear
        # 5. Store num_attention_heads and head_size

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
        # TODO:
        # 1. Project input to query, key, value
        # 2. Reshape Q,K,V to split heads: (batch, num_heads, seq_len, head_size)
        # 3. Calculate attention scores = Q @ K.transpose(-2, -1)
        # 4. Scale scores by 1/sqrt(head_size)
        # 5. Apply softmax and dropout
        # 6. Apply attention to values
        # 7. Reshape and project output
        # 8. Return output and attention weights


class MLP(nn.Module):
    """Multi-Layer Perceptron that processes each patch independently"""
    
    def __init__(self, config):
        """Initialize MLP module.
        
        Args:
            config: Configuration object containing MLP parameters
        """
        super().__init__()
        # TODO:
        # 1. Create fc1 as nn.Linear(hidden_size, intermediate_size)
        # 2. Create activation as nn.GELU()
        # 3. Create fc2 as nn.Linear(intermediate_size, hidden_size)
        # 4. Create dropout layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to input features.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim]
            
        Returns:
            torch.Tensor: Transformed features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        # TODO:
        # 1. Apply fc1, activation, dropout, fc2 in sequence
        # 2. Return transformed features


class EncoderLayer(nn.Module):
    """Single transformer layer combining attention and MLP with residual connections"""
    
    def __init__(self, config: VisionConfig):
        """Initialize a single transformer encoder layer.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        pass

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
        pass


class Encoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, config: VisionConfig):
        """Initialize the transformer encoder with multiple layers.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        pass

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
        pass


class VisionTransformer(nn.Module):
    """Main vision transformer model combining embeddings, encoder, and final layer norm"""
    
    def __init__(self, config: VisionConfig):
        """Initialize the complete vision transformer model.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        pass

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process images through the complete vision transformer pipeline.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape [Batch_Size, Channels, Height, Width]
            
        Returns:
            torch.Tensor: Final encoded features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        pass


class VisionModel(nn.Module):
    """Wrapper class for the vision transformer model"""

    def __init__(self, config: VisionConfig):
        """Initialize the vision model wrapper.
        
        Args:
            config (VisionConfig): Configuration object containing model parameters
        """
        pass

    def forward(self, pixel_values) -> Tuple:
        """Process images through the vision transformer model.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape [Batch_Size, Channels, Height, Width]
            
        Returns:
            torch.Tensor: Encoded image features of shape [Batch_Size, Num_Patches, Embed_Dim]
        """
        pass
