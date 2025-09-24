"""Various ML models classes and functions."""

from copy import deepcopy
from typing import Any, Literal, overload

import torch
from braindecode.models import (
    TCN,
    CTNet,
    Deep4Net,
    EEGConformer,
    EEGInceptionERP,
    EEGNet,
    EEGNetv4,
    EEGNeX,
    ShallowFBCSPNet,
    TIDNet,
)
from mne.decoding import Scaler as MNEScaler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from torch import nn

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.utils.custom_types import ScikitScaler

# A dictionary to easily access braindecode models by string name
BRAINDECODE_MODELS = {
    "EEGNetv4": EEGNetv4,
    "EEGNet": EEGNet,
    "EEGNeX": EEGNeX,
    "ShallowFBCSPNet": ShallowFBCSPNet,
    "Deep4Net": Deep4Net,
    "TIDNet": TIDNet,
    "EEGConformer": EEGConformer,
    "EEGInceptionERP": EEGInceptionERP,
    "TCN": TCN,
    "CTNet": CTNet,
}


def _get_conv_transpose_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int = 0,
    dilation: int = 1,
) -> int:
    """Calculate the output size of a ConvTranspose1d layer."""
    return (
        (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    )


def extract_model_name(metadata: dict[str, Any]) -> str:
    """Extract a clean, human-readable model name from a metadata dictionary.

    This function follows a specific priority order to find the best name:
    1. It first looks for a top-level key named 'model_name'. If found, its
       value is returned directly. This allows for explicit naming.
    2. If not found, it looks for 'model_class_name'.
    3. If that's not found, it falls back to parsing the 'model_class' string,
       which typically looks like "<class 'path.to.ClassName'>". It extracts
       'ClassName' from this string.
    4. If none of the above keys or formats are found, it returns 'Unknown'.

    Example `metadata` inputs and their outputs:
    - {'model_name': 'MyCustomLSTM', ...} -> 'MyCustomLSTM'
    - {'model_class_name': 'LSTMSeq2SeqSimple', ...} -> 'LSTMSeq2SeqSimple'
    - {'model_class': "<class '...models.FlatteningMLP'>", ...} -> 'FlatteningMLP'
    - {'some_other_key': ...} -> 'Unknown'

    Args:
        metadata (dict[str, Any]): The metadata dictionary from a model run.

    Returns:
        str: A clean string representing the model's name.
    """
    # Priority 1: Check for an explicit 'model_name' key.
    if (
        "model_name" in metadata
        and isinstance(metadata["model_name"], str)
        and metadata["model_name"]
    ):
        return metadata["model_name"]

    # Priority 2: Check for 'model_class_name'.
    if (
        "model_class_name" in metadata
        and isinstance(metadata["model_class_name"], str)
        and metadata["model_class_name"]
    ):
        return metadata["model_class_name"]

    # Priority 3: Fallback to parsing the 'model_class' string.
    model_class_str = metadata.get("model_class")
    if isinstance(model_class_str, str):
        # <class 'pretrain_braindecode_models.modeling.models.FlatteningMLP'> -> FlatteningMLP
        model_class_str = model_class_str.strip().replace("<class '", "").replace("'>", "")
        return model_class_str.split(".")[-1]  # Get the last part

    # Priority 4: Return 'Unknown' if no name could be determined.
    return "Unknown"


def count_parameters(model: nn.Module, *, only_trainable: bool = False) -> int:
    """Count the total and/or trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model (nn.Module) to inspect.
        only_trainable (bool, optional): If True, counts only the parameters that require gradients
            (i.e., those that are updated during training). If False, counts all parameters in the
            model. Defaults to False.

    Returns:
        int: The total number of parameters (int).
    """
    total_params = 0

    # Iterate through all parameters of the model
    for param in model.parameters():
        if only_trainable:
            # If we only want trainable, check the `requires_grad` attribute
            if param.requires_grad:
                total_params += param.numel()  # .numel() gives the total number of elements
        else:
            # Otherwise, count all parameters
            total_params += param.numel()

    return total_params


def get_model(model_config: dict[str, Any], data_shape_kwargs: dict[str, int]) -> nn.Module:
    """Instantiate a model from a configuration dictionary.

    Args:
        model_config (dict[str, Any]): A dictionary containing 'model_class' and 'model_kwargs'.
        data_shape_kwargs (dict[str, int]): A dictionary with shape information derived from the
            data, e.g., {'input_features': 64, 'input_seq_len': 1500, ...}.

    Returns:
        nn.Module: An instance of the specified model class with the provided kwargs.
    """
    model_config_copy = deepcopy(model_config)

    model_class_name = model_config_copy["model_class"]
    kwargs = model_config_copy.get("model_kwargs", {})
    additional_kwargs = kwargs.pop("additional_kwargs", {})

    # Combine config kwargs with kwargs derived from data shape
    final_kwargs = {**data_shape_kwargs, **kwargs, **additional_kwargs}

    # Find the model class in the current module's scope
    model_class = globals().get(model_class_name)
    if model_class is None or not issubclass(model_class, nn.Module):
        raise ValueError(
            f"Model class '{model_class_name}' not found or is not a valid nn.Module."
        )

    logger.info(f"Instantiating model '{model_class_name}' with kwargs: {final_kwargs}")

    return model_class(**final_kwargs)


@overload
def get_scaler(
    scaler_name: None,
    scaler_kwargs: dict[str, Any],
) -> None: ...


@overload
def get_scaler(
    scaler_name: str,
    scaler_kwargs: dict[str, Any],
) -> MNEScaler | ScikitScaler: ...


def get_scaler(
    scaler_name: str | None,
    scaler_kwargs: dict[str, Any],
) -> MNEScaler | ScikitScaler | None:
    """Instantiate a scaler by its string name.

    Supported scalers:
        - "MNE": Uses `mne.decoding.Scaler`, which standardizes each channel based on the mean and
          standard deviation computed from the training data.
        - "Standard": Uses `sklearn.preprocessing.StandardScaler`, which standardizes features by
          removing the mean and scaling to unit variance.
        - "MinMaxScaler": Uses `sklearn.preprocessing.MinMaxScaler`, which scales features to a
          given range (default is [0, 1]).
        - "MaxAbs": Uses `sklearn.preprocessing.MaxAbsScaler`, which scales such that the maximal
          absolute value of each feature in the training set will be 1.0.
        - "Robust": Uses `sklearn.preprocessing.RobustScaler`, which scales features using
          statistics that are robust to outliers (e.g., median and interquartile range).
        - None: No scaling is applied.

    Args:
        scaler_name (str | None): The name of the scaler ("MNE", "Standard", "MinMax", "MaxAbs",
            "Robust" or None).
        scaler_kwargs (dict[str, Any]): Additional keyword arguments for the scaler.

    Returns:
        (MNEScaler | ScikitScaler): An instance of the specified scaler, or None if no scaling
            is to be applied.
    """
    if scaler_name is None:
        return None

    name_lower = scaler_name.lower()
    if name_lower == "mne":
        return MNEScaler(**scaler_kwargs)
    if name_lower == "standard":
        return StandardScaler(**scaler_kwargs)
    if name_lower == "minmax":
        return MinMaxScaler(**scaler_kwargs)
    if name_lower == "maxabs":
        return MaxAbsScaler(**scaler_kwargs)
    if name_lower == "robust":
        return RobustScaler(**scaler_kwargs)

    raise NotImplementedError(
        f"Unknown scaler type: '{scaler_name}'. Choose from 'MNE', 'Standard', or None."
    )


class SelectLastTimestep(nn.Module):
    """A PyTorch module that selects the last timestep from a 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last timestep of a 3D tensor.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, n_channels, n_times)`

        Returns:
            torch.Tensor: Tensor of shape `(batch, n_channels)`
        """
        # Selects all batches, all channels, and the last time entry
        return x[:, :, -1]


class BraindecodeEncoder(nn.Module):
    """Pre-trained encoder from [braindecode](https://braindecode.org/stable/index.html) library.

    This class uses a Braindecode model (e.g., EEGNetv4, TCN, ShallowFBCSPNet) as a
    feature extractor (the "encoder"). It then attaches a custom "decoder" on top to
    transform the learned EEG features into a sequence of FLAME parameters for 3D
    face animation.

    - Input Shape: `(batch, n_channels, n_timesteps)` -> e.g., `(32, 64, 1500)`
    - Output Shape: `(batch, output_features, output_seq_len)` -> e.g., `(32, 120, 29)`

    The model is designed to be highly adaptable, supporting two main types of decoders
    controlled by the `decoder_type` parameter.

    Decoder Architectures (`decoder_type`):
    --------------------------------------
    1. `decoder_type="mlp"` (Flattening MLP Decoder):
        - **How it works:** This is the simplest approach. The feature map from the
          encoder (which can be 2D, 3D, or 4D) is completely flattened into a single
          long vector for each item in the batch. This vector is then fed through
          a standard Multi-Layer Perceptron (MLP) with Linear layers, normalization,
          and dropout to produce a flattened output vector, which is then reshaped
          to the target sequence shape `(batch, output_features, output_seq_len)`.
        - **Best for:** Encoders that produce a single feature vector per input window
          (e.g., Deep4Net, EEGResNet, TIDNet after their pooling layers). It acts as a
          powerful classifier/regressor on top of the learned features.
        - **Pros:** Simple, robust, and works with any encoder output shape.
        - **Cons:** Discards all temporal structure present in the encoder's feature map.

    2. `decoder_type="temporal_conv"` (Temporal Convolutional Decoder):
        - **How it works:** This decoder is designed to preserve, transform, and reshape
          the temporal dimension of the encoder's output. It handles various encoder
          outputs intelligently:
            - **4D/3D Tensors (e.g., EEGNetv4, ShallowFBCSPNet):** It reshapes the
              feature map to a 3D tensor of shape `(batch, channels, time)`.
            - **2D Tensors (e.g., TIDNet, EEGInceptionERP):** It unsqueezes the feature
              vector to `(batch, channels, 1)`, treating it as a sequence of length 1.
          It then uses a series of `ConvTranspose1d` layers to learn an upsampling
          of the temporal sequence until its length is >= `output_seq_len`. A final
          `Conv1d` maps to the correct number of `output_features`. A final, non-learned
          interpolation step ensures the exact `output_seq_len`.
        - **Best for:** Any CNN-based encoder where you want to maintain the temporal
          relationship between the learned features.
        - **Pros:** Learns a powerful temporal transformation. More suitable for
          sequence-to-sequence tasks than the flattening MLP.
        - **Cons:** More complex than the MLP.

    Special Handling for TCN Model:
    -------------------------------
    The TCN model from Braindecode is a special case. It's already a sequence-to-sequence
    model that produces an output sequence of the same length as the input. To adapt this
    to our (potentially different) `output_seq_len`, this class applies a specific
    decoder when `braindecode_model_name="TCN"`:
    1.  The TCN encoder's output `(batch, features, input_seq_len)` is passed to
        `nn.AdaptiveAvgPool1d(output_seq_len)` to downsample the temporal dimension
        in a learnable-aware way.
    2.  A final `nn.Conv1d` with `kernel_size=1` maps the feature channels to the
        desired `output_features`.
    This is much more efficient and effective than interpolating over a large time gap.
    This is applied if `tcn_special_handling` is enabled and `braindecode_model_name="TCN"`.
    """

    def __init__(
        self,
        input_features: int,  # n_channels
        input_seq_len: int,  # n_timesteps
        output_features: int,
        output_seq_len: int,
        *,
        braindecode_model_name: str = "EEGNetv4",
        decoder_type: Literal["mlp", "temporal_conv"] = "mlp",
        decoder_param_shapes: dict[str, int] | None = None,
        mlp_hidden_dims: list[int] = [1024, 512],
        mlp_dropout_prob: float = 0.4,
        freeze_encoder: bool = True,
        name: str | None = "BraindecodeEncoder",
        tcn_special_handling: bool = True,
        **kwargs,  # Additional kwargs for the braindecode model
    ) -> None:
        """Initialize the BraindecodeEncoder model.

        Args:
            input_features (int): Number of EEG channels.
            input_seq_len (int): Number of time samples in the EEG window.
            output_features (int): Number of features in the output sequence
                (e.g., 120 FLAME params).
            output_seq_len (int): The length of the output time sequence (e.g., 30 frames).
            braindecode_model_name (str): The name of the braindecode model to use.
                Must be one of: EEGNetv4, ShallowFBCSPNet, etc.
            decoder_type (Literal["mlp", "temporal_conv"]): Type of decoder to use.
            decoder_param_shapes (dict[str, int] | None): If provided, creates a separate
                decoder head for each key-value pair, where the key is the parameter group
                name and the value is its feature dimension. The `output_features` argument
                is ignored in this case. Defaults to None (single decoder).
            mlp_hidden_dims (list[int]): A list of hidden layer sizes for the decoder MLP.
            mlp_dropout_prob (float): Dropout probability for the decoder MLP.
            freeze_encoder (bool): If True, the weights of the pre-trained braindecode
                model will be frozen and not updated during training.
            name (str | None): Optional name for the model.
            tcn_special_handling (bool): If True, applies special handling for TCN models.
            **kwargs: Additional keyword arguments for the braindecode model.
                These will be passed to the model's constructor.
        """
        super().__init__()

        # --- Store all __init__ args for introspection ---
        self.input_features = input_features
        self.input_seq_len = input_seq_len
        self.output_features = output_features
        self.output_seq_len = output_seq_len
        self.braindecode_model_name = braindecode_model_name
        self.decoder_type = decoder_type
        self.decoder_param_shapes = decoder_param_shapes
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout_prob = mlp_dropout_prob
        self.freeze_encoder = freeze_encoder
        self.kwargs = kwargs
        self.name = name
        self.tcn_special_handling = tcn_special_handling
        self._use_reshape = False  # Flag to track if we had to use reshape instead of view

        if braindecode_model_name not in BRAINDECODE_MODELS:
            raise ValueError(
                f"Unknown braindecode model '{braindecode_model_name}'. "
                f"Choose from {list(BRAINDECODE_MODELS.keys())}"
            )

        # --- Load Pre-trained Braindecode Model ---
        # Note: Braindecode models require n_channels and n_times.
        # We also pass `n_classes=1` as a placeholder, as we'll remove the final layer, since we
        # aren't doing classification
        EncoderClass = BRAINDECODE_MODELS[braindecode_model_name]

        if braindecode_model_name == "TCN":
            self.encoder = EncoderClass(
                n_chans=input_features,
                n_outputs=output_features,  # In TCN, (1, 64, 500) -> (1, 120, 444)
                **self.kwargs,  # Additional kwargs for the braindecode model
            )
        else:
            self.encoder = EncoderClass(
                n_chans=input_features,
                n_times=input_seq_len,
                # Number of outputs of the model. This is the number of classes in the case of
                # classification.
                n_outputs=1,  # Placeholder, will be ignored
                **self.kwargs,  # Additional kwargs for the braindecode model
            )

        # --- Correctly separate the feature extractor from the classifier head ---
        # We will create a new Sequential module for the feature extractor part
        # and store the classifier head separately (though we won't use it).
        if braindecode_model_name in [
            "EEGNetv4",
            "EEGNet",
            "EEGNeX",
            "ShallowFBCSPNet",
            "Deep4Net",
            "TIDNet",
            "EEGInceptionERP",
            "CTNet",
        ]:
            # For these models, the classifier is the last module in the sequence.
            self.encoder.final_layer = nn.Identity()  # Replace the final layer with Identity

            if hasattr(self.encoder, "final_conv"):
                self.encoder.final_conv = nn.Identity()

            # self.classifier_head = encoder_modules[-1]  # We store it but won't use it in forward
            logger.info(
                f"Separated feature extractor from final layer of {braindecode_model_name}."
            )

        elif braindecode_model_name == "EEGResNet":
            # For EEGResNet, the classifier is the `fc` layer.
            # We take all modules except the fc layer.
            # self.classifier_head = self.encoder.fc
            self.encoder.final_layer = nn.Identity()  # Replace the final layer with Identity
            self.encoder.fc = nn.Identity()  # Replace it in the original model
            logger.info(
                f"Separated feature extractor from `fc` layer of {braindecode_model_name}."
            )

        elif braindecode_model_name == "EEGConformer":
            self.encoder = EncoderClass(
                n_chans=input_features,
                n_times=input_seq_len,
                return_features=True,  # This will return the feature map directly
                n_outputs=1,  # Placeholder, will be ignored
                **self.kwargs,  # Additional kwargs for the braindecode model
            )
            logger.info(
                f"Initialized {braindecode_model_name} with return_features=True to get the "
                "feature map directly."
            )

        # For TCN, we don't have a final layer to remove, so we skip this
        elif braindecode_model_name != "TCN":
            raise NotImplementedError(
                f"Head removal for {braindecode_model_name} is not implemented."
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Froze {braindecode_model_name} encoder.")

        # --- 2. Determine the size of the FLATTENED feature vector ---
        with torch.no_grad():
            self.encoder.eval()
            dummy_input = torch.zeros(1, input_features, input_seq_len)
            feature_map = self.encoder(dummy_input)

        logger.info(f"Initial Braindecode feature map shape: {feature_map.shape}")

        # --- Multi-head vs. Single-head logic ---
        if self.decoder_param_shapes:
            logger.info(
                "Initializing multi-head decoder for parameters: "
                f"{list(self.decoder_param_shapes.keys())}"
            )
            if sum(self.decoder_param_shapes.values()) != self.output_features:
                logger.warning(
                    f"Sum of decoder_param_shapes ({sum(self.decoder_param_shapes.values())}) "
                    f"does not match output_features ({self.output_features}). Using the sum."
                )

            # Create a dictionary of decoder heads
            self.decoders = nn.ModuleDict()
            for param_name, param_dim in self.decoder_param_shapes.items():
                self.decoders[param_name] = self._create_decoder_head(
                    encoder_feature_map=feature_map,
                    head_output_features=param_dim,
                    head_output_seq_len=self.output_seq_len,
                )
                decoder_parts = [s.strip() for s in str(self.decoders[param_name]).split("\n")]
                logger.debug(f"Created decoder for '{param_name}': {' '.join(decoder_parts)}")
        else:
            # Original single-head logic
            logger.info("Initializing single monolithic decoder.")
            self.decoder = self._create_decoder_head(
                encoder_feature_map=feature_map,
                head_output_features=self.output_features,
                head_output_seq_len=self.output_seq_len,
            )
            # Add to a ModuleDict for consistent forward pass logic
            self.decoders = nn.ModuleDict({"full_output": self.decoder})

    def _create_decoder_head(
        self,
        encoder_feature_map: torch.Tensor,
        head_output_features: int,
        head_output_seq_len: int,
    ) -> nn.Module:
        """Create a single decoder head."""
        feature_map = encoder_feature_map

        # --- Handle TCN as a Special Case ---
        if self.braindecode_model_name == "TCN" and self.tcn_special_handling:
            encoder_out_channels = feature_map.shape[1]
            decoder = nn.Sequential(
                nn.AdaptiveAvgPool1d(head_output_seq_len),
                nn.Conv1d(encoder_out_channels, head_output_features, kernel_size=1),
            )
            logger.debug(
                "Built a TCN-specific decoder with "
                f"AdaptiveAvgPool1d({feature_map.shape}) -> "
                f"Conv1d({encoder_out_channels}, {head_output_features}, 1)"
            )
            return decoder

        # --- Pre-process the feature map to a consistent shape for decoders ---
        pre_processor: nn.Module = nn.Identity()
        if feature_map.ndim == 4:
            # For models like EEGNetv4 (B, C, 1, T) or ShallowFBCSPNet (B, C, T, 1)
            # We want to flatten the last two dimensions to get (B, C, T_new)
            logger.info("Encoder output is 4D. Adding a pre-processor to reshape to 3D.")

            class Reshape4Dto3D(nn.Module):
                """Reshape 4D feature maps to 3D with a view operation."""

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x.view(x.shape[0], x.shape[1], -1)

            pre_processor = Reshape4Dto3D()
            # Update feature_map shape for decoder construction
            feature_map = pre_processor(feature_map)
            logger.debug(f"Reshaped feature map shape for decoder: {feature_map.shape}")

        # TIDNet produces a feature_map of shape (batch_size, 2640)
        # So reshape it to have (batch_size, features, 1)
        elif feature_map.ndim == 2 and self.decoder_type == "temporal_conv":

            class Reshape2Dto3D(nn.Module):
                """Reshape 2D feature maps to 3D by unsqueezing a temporal dimension."""

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x.unsqueeze(-1)  # Add a temporal dimension of size 1

            pre_processor = Reshape2Dto3D()
            feature_map = pre_processor(feature_map)
            logger.warning(
                f"Encoder output is 2D. Reshaping to 3D with length 1 for temporal "
                f"decoder: {feature_map.shape}"
            )

        if self.decoder_type == "temporal_conv" and feature_map.ndim != 3:
            raise ValueError(
                f"Encoder '{self.braindecode_model_name}' produces a {feature_map.ndim}D feature "
                "map, which is incompatible with the 'temporal_conv' decoder. "
            )

        # --- Build the Appropriate Decoder based on requested type and feature map shape ---
        decoder_layers: list[nn.Module] = []
        if self.decoder_type == "temporal_conv":
            # --- Convolutional Decoder (Transposed Convs for Upsampling) ---
            encoder_out_channels = feature_map.shape[1]
            current_len = feature_map.shape[2]
            current_channels = encoder_out_channels

            # Build a dynamic upsampling path
            while current_len < head_output_seq_len:
                # Double the sequence length in each step
                stride = 2
                # Choose a kernel size that doesn't overshoot the target too much
                kernel_size = 4
                padding = 1

                # Determine number of output channels for this layer
                out_channels = max(head_output_features, current_channels // 2)

                tconv = nn.ConvTranspose1d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                decoder_layers.append(tconv)
                decoder_layers.append(nn.ReLU())

                current_channels = out_channels
                current_len = _get_conv_transpose_output_size(
                    current_len, kernel_size, stride, padding
                )

            # Final 1x1 conv to match the exact output feature dimension
            decoder_layers.append(nn.Conv1d(current_channels, head_output_features, kernel_size=1))

        elif self.decoder_type == "mlp":
            # --- Flattening MLP Decoder (for standard CNNs) ---
            # Add a Flatten layer: This will be the first layer in the sequence for MLP decoders
            decoder_layers.append(nn.Flatten())

            encoder_output_size = feature_map.reshape(1, -1).shape[1]
            output_flat = head_output_seq_len * head_output_features
            current_dim = encoder_output_size

            for h_dim in self.mlp_hidden_dims:
                decoder_layers.extend(
                    [
                        nn.Linear(current_dim, h_dim),
                        nn.LayerNorm(h_dim),  # LayerNorm is often good here
                        nn.ReLU(),
                        nn.Dropout(self.mlp_dropout_prob),
                    ]
                )
                current_dim = h_dim

            decoder_layers.append(nn.Linear(current_dim, output_flat))

        # Do not add the preprocessor if it's just Identity
        if isinstance(pre_processor, nn.Identity):
            return nn.Sequential(*decoder_layers)

        return nn.Sequential(pre_processor, *decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Input x shape: (batch, n_channels, n_timesteps) - as required by braindecode

        # Get EEG features from the encoder
        feature_map = self.encoder(x)  # -> output shape is something like (batch, 16, 1, 15)

        # Process the feature map through each decoder head
        head_outputs = []
        for param_name, decoder_head in self.decoders.items():
            head_output = decoder_head(feature_map)

            # --- Reshape MLP output if necessary ---
            # The MLP head outputs a flat vector which we reshape right away
            if self.decoder_type == "mlp" or (
                self.braindecode_model_name == "TCN" and not self.tcn_special_handling
            ):
                output_dim = (
                    self.decoder_param_shapes[param_name]
                    if self.decoder_param_shapes
                    else self.output_features
                )
                if self._use_reshape:
                    head_output = head_output.reshape(-1, output_dim, self.output_seq_len)
                else:
                    try:
                        # Try to use view() for better performance
                        head_output = head_output.view(-1, output_dim, self.output_seq_len)
                    except RuntimeError:
                        # Sometimes view() fails if the tensor is not contiguous.
                        # For example, RuntimeError: view size is not compatible with input
                        # tensor's size and stride (at least one dimension spans across two
                        # contiguous subspaces). Use .reshape(...) instead.
                        head_output = head_output.reshape(-1, output_dim, self.output_seq_len)
                        self._use_reshape = True

            # Ensure correct sequence length for temporal conv decoders via interpolation
            if (
                self.decoder_type == "temporal_conv"
                and head_output.shape[2] != self.output_seq_len
            ):
                # logger.warning(
                #     f"Conv decoder output length ({output_sequence.shape[2]}) does not match "
                #     f"target ({self.output_seq_len}). Resizing with interpolation as a fallback."
                # )
                head_output = nn.functional.interpolate(
                    head_output,
                    size=self.output_seq_len,
                    mode="linear",
                    align_corners=False,
                )

            head_outputs.append(head_output)

        # Concatenate the outputs from all heads along the feature dimension
        if len(head_outputs) > 1:
            return torch.cat(head_outputs, dim=1)

        return head_outputs[0]


class BraindecodeClassifier(nn.Module):
    """Classifier from [braindecode](https://braindecode.org/stable/index.html) library.

    This class wraps a pre-trained Braindecode model (e.g., EEGNetv4, TCN, ShallowFBCSPNet)
    and provides a simple interface for classification tasks.

    Special Handling for TCN Model:
    -------------------------------
    The TCN model from Braindecode is a special case. It's already a sequence-to-sequence
    model that produces an output sequence of the same length as the input. This wrapper
    automatically aggregates these temporal predictions by doing one of these 3 strategies:

    1. **Use the Last Timestep**: This is a very common approach. The idea is that the final
    timestep of the TCN has the largest receptive field and contains the most comprehensive
    information about the entire input sequence. You simply select the predictions at the last
    time point.
    2. **Global Average Pooling**: Take the average of the predictions across the entire time
    dimension. This considers information from all timesteps equally.
    3. **Global Max Pooling**: Take the maximum value for each class across the time dimension.
    This focuses on the point in time where the model was "most confident" about a particular
    class.
    """

    def __init__(
        self,
        input_features: int,  # n_channels
        input_seq_len: int,  # n_timesteps
        n_outputs: int,  # Number of classes for classification
        *,
        braindecode_model_name: str = "EEGNetv4",
        freeze_encoder: bool = False,
        tcn_pooling_strategy: Literal["last", "mean", "max"] = "last",
        name: str | None = "BraindecodeClassifier",
        **kwargs,  # Additional kwargs for the braindecode model
    ) -> None:
        """Initialize the BraindecodeClassifier model.

        Args:
            input_features (int): Number of EEG channels.
            input_seq_len (int): Number of time samples in the EEG window.
            n_outputs (int): Number of outputs of the model. This is the number of classes in the
                case of classification.
            braindecode_model_name (str): The name of the braindecode model to use.
                Must be one of: EEGNetv4, ShallowFBCSPNet, etc.
            freeze_encoder (bool): If True, the weights of the pre-trained braindecode
                model will be frozen and not updated during training.
            tcn_pooling_strategy (Literal["last", "mean", "max"]): The strategy to aggregate
                temporal predictions when using the TCN model:
                - 'last': Use the last time step's prediction.
                - 'mean': Global Average Pooling. Use the average prediction across time.
                - 'max': Global Max Pooling. Use the max prediction across time.
            name (str | None): Optional name for the model.
            **kwargs: Additional keyword arguments for the braindecode model.
                These will be passed to the model's constructor.
        """
        super().__init__()

        # --- Store all __init__ args for introspection ---
        self.input_features = input_features
        self.input_seq_len = input_seq_len
        self.n_outputs = n_outputs
        self.braindecode_model_name = braindecode_model_name
        self.freeze_encoder = freeze_encoder
        self.tcn_pooling_strategy = tcn_pooling_strategy
        self.kwargs = kwargs
        self.name = name

        if self.tcn_pooling_strategy not in ["last", "mean", "max"]:
            raise ValueError(
                f"Invalid tcn_pooling_strategy '{self.tcn_pooling_strategy}'. "
                "Choose from 'last', 'mean', or 'max'."
            )

        # This will hold the main model
        self.model: nn.Module

        # This will hold the aggregation layer for sequence models like TCN
        self.aggregator: nn.Module = nn.Identity()

        if braindecode_model_name not in BRAINDECODE_MODELS:
            raise ValueError(
                f"Unknown braindecode model '{braindecode_model_name}'. "
                f"Choose from {list(BRAINDECODE_MODELS.keys())}"
            )

        # --- Load Pre-trained Braindecode Model ---
        # Note: Braindecode models require n_channels and n_times.
        EncoderClass = BRAINDECODE_MODELS[braindecode_model_name]

        if braindecode_model_name == "TCN":
            self.model = EncoderClass(
                n_chans=input_features,
                n_outputs=n_outputs,
                **self.kwargs,  # Additional kwargs for the braindecode model
            )

            # --- Create the aggregator layer based on the strategy ---
            if tcn_pooling_strategy == "last":
                self.aggregator = SelectLastTimestep()
            elif tcn_pooling_strategy == "mean":
                self.aggregator = nn.Sequential(nn.AdaptiveAvgPool1d(output_size=1), nn.Flatten())
            elif tcn_pooling_strategy == "max":
                self.aggregator = nn.Sequential(nn.AdaptiveMaxPool1d(output_size=1), nn.Flatten())
            logger.info(f"Initialized TCN with '{tcn_pooling_strategy}' pooling aggregator.")
        else:
            self.model = EncoderClass(
                n_chans=input_features,
                n_times=input_seq_len,
                n_outputs=n_outputs,
                **self.kwargs,  # Additional kwargs for the braindecode model
            )

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info(f"Froze weights of pre-trained {braindecode_model_name} encoder.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Get the output from the core model
        # For TCN, this is a sequence (B, C, T)
        # For others, it's a vector (B, C)
        output = self.model(x)

        # Apply the aggregator
        # For TCN, this will perform the pooling (last, mean, or max).
        # For other models, this is nn.Identity() and does nothing.
        return self.aggregator(output)


class FlatteningMLP(nn.Module):
    """A simple MLP that flattens the input and output.

    This model is designed to take a sequence of EEG data and output a sequence of FLAME params.
    - The input shape is expected to be `(batch, seq_len_in, features_in)`,
    - The output shape is `(batch, seq_len_out, features_out)`.
    """

    def __init__(
        self,
        input_seq_len: int,
        input_features: int,
        output_seq_len: int,
        output_features: int,
        *,
        name: str | None = "MLP",
        hidden_dims: list[int] = [512, 256, 128, 64],
        dropout_prob: float | None = 0.3,
        norm_layer: Literal["batch", "layer"] | None = "layer",
    ) -> None:
        """Initialize the FlatteningMLP model.

        Args:
            input_seq_len: The length of the input sequence.
            input_features: Number of features in each input step.
            output_seq_len: The length of the output sequence.
            output_features: Number of features in each output step.
            name: Optional name for the model.
            hidden_dims: A list of hidden layer sizes for the MLP.
            dropout_prob: Dropout probability.
            norm_layer: The type of normalization to use. Can be "batch" for
                        BatchNorm1d, "layer" for LayerNorm, or None for no
                        normalization. Defaults to "batch".
        """
        super().__init__()

        # Store parameters
        self.input_seq_len = input_seq_len
        self.input_features = input_features
        self.output_seq_len = output_seq_len
        self.output_features = output_features
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        self.norm_layer = norm_layer
        self.name = name

        # Calculate flattened input and output sizes
        input_flat = input_seq_len * input_features
        output_flat = output_seq_len * output_features
        self.output_shape = (output_features, output_seq_len)

        # Build the Network
        layers: list[nn.Module] = []
        current_dim = input_flat

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))

            # Normalization is applied on the raw linear output (before activation)
            if self.norm_layer == "batch":
                # BatchNorm1d normalizes across the batch for each feature.
                # It expects the channel/feature dimension as input.
                layers.append(nn.BatchNorm1d(h_dim))
            elif self.norm_layer == "layer":
                # LayerNorm normalizes across the features for each item in the batch.
                # It expects the shape of the features to normalize.
                layers.append(nn.LayerNorm(h_dim))

            # If norm_layer is None, no normalization layer is added.
            layers.append(nn.ReLU())  # Use ReLU for non-linearity

            # Dropout is applied after activation
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(self.dropout_prob))

            current_dim = h_dim

        # Final output layer (no batch norm or activation after this)
        layers.append(nn.Linear(current_dim, output_flat))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # x shape: (batch, seq_len, features)
        # Use .reshape() instead of .view() to handle non-contiguous tensors
        x = x.reshape(x.size(0), -1)  # After flattening: (batch, input_seq_len * input_features)

        # Pass through the MLP
        x = self.net(x)

        # Reshape output to (batch, seq_len_out, features_out)
        return x.view(x.size(0), *self.output_shape)


class LSTMSeq2SeqSimple(nn.Module):
    """A simple LSTM-based sequence-to-sequence model for regression.

    It uses an LSTM to encode the input sequence into a context vector,
    which is then passed through a multi-layer perceptron (MLP) to generate
    the entire output sequence at once.

    - Input Shape: `(batch_size, input_features, input_seq_len)` -> e.g., `(32, 64, 1500)`
    - Output Shape: `(batch_size, output_features, output_seq_len)` -> e.g., `(32, 120, 30)`
    """

    def __init__(
        self,
        input_features: int,
        output_seq_len: int,
        output_features: int,
        *,
        input_seq_len: int | None = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        mlp_hidden_dims: list[int] = [256, 512],
        mlp_dropout_prob: float = 0.5,
        mlp_norm_layer: Literal["batch", "layer"] | None = "layer",
        lstm_dropout_prob: float = 0.3,
        name: str | None = "LSTM",
        bidirectional: bool = True,
    ) -> None:
        """Initialize the LSTMSeq2SeqSimple model.

        Args:
            input_features (int): The number of features in each step of the input sequence
                (e.g., 64 EEG channels).
            input_seq_len (int): The length of the input sequence (e.g., 1500 time steps).
                **Note:** This parameter is unused, it's only put here to maintain consistency
                with other models.
            output_seq_len (int): The length of the output sequence (e.g., 30 FLAME frames).
            output_features (int): The number of features in each step of the output sequence
                (e.g., 120 FLAME params).
            lstm_hidden_size (int): The size of the LSTM's hidden state.
            lstm_num_layers (int): The number of stacked LSTM layers.
            mlp_hidden_dims (list[int]): A list of hidden layer sizes for the decoder MLP.
            mlp_dropout_prob (float): Dropout probability for regularization in the MLP decoder.
            mlp_norm_layer (Literal["batch", "layer"] | None): The type of normalization to use for
                the MLP decoder.
            lstm_dropout_prob (float): Dropout probability for regularization in the LSTM.
            name (str | None): Optional name for the model.
            bidirectional (bool): If True, uses a bidirectional LSTM. Defaults to True
        """
        super().__init__()

        # Store parameters
        self.input_features = input_features
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.output_features = output_features
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout_prob = mlp_dropout_prob
        self.mlp_norm_layer = mlp_norm_layer
        self.lstm_dropout_prob = lstm_dropout_prob
        self.name = name
        self.bidirectional = bidirectional

        # --- LSTM Encoder ---
        # Takes the input sequence and processes it. batch_first=True is crucial.
        # Bidirectional means it processes the sequence forwards and backwards.
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout_prob
            if lstm_num_layers > 1
            else 0,  # Dropout is only between LSTM layers
            bidirectional=bidirectional,
        )

        # The context vector will be the concatenated final hidden states of the
        # forward and backward LSTMs. So its size is 2 * lstm_hidden_size
        encoder_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size

        # --- MLP Decoder ---
        # This part takes the context vector and maps it to the desired output sequence shape.
        decoder_layers: list[nn.Module] = []
        input_dim = encoder_output_size

        for hidden_dim in mlp_hidden_dims:
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))

            # Normalization is applied on the raw linear output (before activation)
            if self.mlp_norm_layer == "batch":
                # BatchNorm1d normalizes across the batch for each feature.
                # It expects the channel/feature dimension as input.
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.mlp_norm_layer == "layer":
                # LayerNorm normalizes across the features for each item in the batch.
                # It expects the shape of the features to normalize.
                decoder_layers.append(nn.LayerNorm(hidden_dim))

            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(mlp_dropout_prob))
            input_dim = hidden_dim  # For the next layer

        # Final layer to produce the flattened output sequence
        decoder_layers.append(nn.Linear(input_dim, output_seq_len * output_features))

        self.decoder_mlp = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, input_features, input_seq_len)`.

        Returns:
            (torch.Tensor): Output tensor of shape `(batch, output_features, output_seq_len)`.
        """
        # --- Transpose input for LSTM ---
        # (batch, features, time) -> (batch, time, features)
        x = x.permute(0, 2, 1)
        # x shape is now: (batch, 1500, 64)

        # --- Encoding Step ---
        # We don't need the output of every time step, just the final hidden states.
        # `_` will hold all hidden states, `(h_n, c_n)` will hold the final ones.
        _, (hidden, cell) = self.lstm(x)

        # `hidden` shape: (num_layers * num_directions, batch, lstm_hidden_size)
        # e.g., (2 * 2, batch, 128) -> (4, batch, 128)

        # We want the final hidden state from the last layer of both directions.
        # For a 2-layer bidirectional LSTM, hidden states are ordered:
        # [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer_1]
        # We grab the last forward (-2) and last backward (-1) states.
        if self.bidirectional:
            last_forward_hidden = hidden[-2, :, :]
            last_backward_hidden = hidden[-1, :, :]

            # Concatenate them to form the final context vector
            context_vector = torch.cat((last_forward_hidden, last_backward_hidden), dim=1)
            # context_vector shape: (batch, lstm_hidden_size * 2)
        else:
            # If not bidirectional, just take the last hidden state
            last_forward_hidden = hidden[-1, :, :]
            context_vector = last_forward_hidden
            # context_vector shape: (batch, lstm_hidden_size)

        # --- Decoding Step ---
        # Pass the context vector through the MLP decoder
        flat_output = self.decoder_mlp(context_vector)
        # flat_output shape: (batch, output_seq_len * output_features)

        # Reshape the flat output into the desired sequence format
        output_sequence_ntc = flat_output.view(-1, self.output_seq_len, self.output_features)
        # output_sequence shape: (batch, 30, 120)

        # --- Transpose output to the desired NCT format ---
        # (batch, time, features) -> (batch, features, time)
        output_sequence_nct = output_sequence_ntc.permute(0, 2, 1)
        # output_sequence_nct shape: (batch, 120, 30)

        return output_sequence_nct  # noqa: RET504
