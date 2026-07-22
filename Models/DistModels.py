"""
Neural network models for learning distance metrics.
Includes MAD and TD-MAD encoders with configurable distance functions.
"""

import torch
from DistExpReplay.ErDist import ErDist
import torchqmet
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


class RunningMeanStdNorm(nn.Module):
    """
    Tracks running mean and standard deviation to normalize inputs (standardization).
    Uses Welford's algorithm for numerically stable online updates.
    """
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.training:
            # Update running stats using Welford's algorithm within no_grad
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False) # Population variance for the batch
                batch_count = x.size(0)

                if batch_count == 0:
                    # Avoid division by zero if batch is empty
                    return (x - self.running_mean) / (torch.sqrt(self.running_var.clamp(min=self.eps)))

                delta = batch_mean - self.running_mean
                tot_count = self.count + batch_count

                new_mean = self.running_mean + delta * batch_count / tot_count
                
                m_a = self.running_var * self.count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
                
                new_var = M2 / tot_count
                
                self.running_mean.copy_(new_mean)
                self.running_var.copy_(new_var)
                self.count.copy_(tot_count)

        # Normalize using the running stats
        return (x - self.running_mean) / torch.sqrt(self.running_var.clamp(min=self.eps))


class ResidualBlock(nn.Module):
    """
    A residual block that implements skip connections in the neural network.
    
    This block adds the input directly to the output of a sequence of layers,
    helping with gradient flow and feature preservation in deep networks.
    
    Attributes:
        block (nn.Sequential): Sequence of layers in the residual block
    """

    def __init__(self, dim: int, dropout_rate: float = 0.1) -> None:
        """
        Initialize the residual block.

        Args:
            dim (int): Input/output dimension of the block
            dropout_rate (float): Dropout probability for regularization
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        identity = x  # Store original input
        out = self.block(x)  # Process through the block
        return out + identity  # Add the original input to the output


class Encoder(nn.Module):
    """
    A neural network encoder that maps input states to a latent space.
    
    This encoder can be configured with various architectural choices including
    layer normalization, residual connections, and positive output forcing.
    
    Attributes:
        feature_size (int): Size of the last hidden layer
        force_positive (bool): Whether to force positive outputs
        layers (nn.Sequential): Main network layers
        final_layer (nn.Linear): Final output layer
    """

    def __init__(
            self,
            in_d: int,
            out_d: int,
            hidden_dims: List[int] = [512, 512, 256, 128],
            use_layer_norm: bool = False,
            dropout_rate: float = 0.,
            use_residual: bool = False,
            force_positive: bool = False,
            min_max_norm: bool = True,
            use_weight_norm: bool = False,
    ) -> None:
        """
        Initialize the encoder network.

        Args:
            in_d (int): Input dimension
            out_d (int): Output dimension
            hidden_dims (List[int]): List of hidden layer dimensions
            use_layer_norm (bool): Whether to use layer normalization
            dropout_rate (float): Dropout probability for regularization
            use_residual (bool): Whether to use residual connections
            force_positive (bool): Whether to force positive outputs
        """
        super().__init__()

        layers = []

        if min_max_norm:
            layers.append(RunningMeanStdNorm(in_d))

        current_dim = in_d

        self.feature_size = hidden_dims[-1]
        self.force_positive = force_positive


        # Create the main network layers
        for hidden_dim in hidden_dims:

            linear_layer = nn.Linear(current_dim, hidden_dim)
            if use_weight_norm:
                linear_layer = nn.utils.weight_norm(linear_layer)

            # First add dimension-changing block
            if use_layer_norm:
                layers.extend([
                    linear_layer,
                    nn.LayerNorm(hidden_dim),
                    nn.SELU(),
                    nn.Dropout(dropout_rate)
                ])
            else:
                layers.extend([
                    linear_layer,
                    nn.SELU(),
                    nn.Dropout(dropout_rate)
                ])

            # Then add residual block if enabled and dimensions allow
            if use_residual:
                layers.append(ResidualBlock(hidden_dim, dropout_rate))

            current_dim = hidden_dim

        # Final output layer
        self.final_layer = nn.Linear(current_dim, out_d)
        if use_weight_norm:
            self.final_layer = nn.utils.weight_norm(self.final_layer)

        # Package all layers
        self.layers = nn.Sequential(*layers)

        self.out_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize network weights.
        Uses LeCun Normal initialization for layers followed by SELU, as this is
        required to maintain the self-normalizing properties of SELU.
        Other layers use standard Kaiming or Xavier initialization.
        This method correctly handles both standard and weight-normalized layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize the underlying weights of the linear layer
                self._initialize_linear_layer(module)
                
                # If the layer is weight-normalized, also initialize the magnitude 'g'
                if hasattr(module, 'weight_g'):
                    nn.init.ones_(module.weight_g)

    def _initialize_linear_layer(self, module: nn.Linear):
        """Helper function to initialize a single linear layer's weights."""
        # Determine if this is the final layer.
        # This check works whether the final_layer is wrapped in weight_norm or not.
        is_final = (module is getattr(self.final_layer, 'module', self.final_layer))
        
        # The weight parameter to initialize is 'weight_v' for normalized layers,
        # and 'weight' for standard layers.
        weight_param = module.weight_v if hasattr(module, 'weight_v') else module.weight

        if not is_final:
            # Use LeCun Normal initialization for SELU in hidden layers
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_param)
            std = (1 / fan_in) ** 0.5
            nn.init.normal_(weight_param, mean=0.0, std=std)
        else:
            # For the final layer, use a different scheme
            if self.force_positive:
                # Kaiming for ReLU output
                nn.init.kaiming_uniform_(weight_param, nonlinearity="relu")
            else:
                # Kaiming for linear output
                nn.init.kaiming_uniform_(weight_param, nonlinearity="linear")
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        x = self.layers(x)
        x = self.final_layer(x)
        if self.force_positive:
            x = F.relu(x)
        return x * self.out_alpha


class MadDistEncoder(torch.nn.Module):
    """
    Neural network model for learning the Minimum Action Distance (MAD) embedding.
    
    This model learns a distance function between states in a latent space,
    using various distance metrics (WideNorm, L1, IQE, or Simple).
    
    Attributes:
        encoder (Encoder): Neural network encoder
        dist_type (str): Type of distance metric to use
        param_dist (Optional[torch.nn.Module]): Parameterized distance function
        counters (Dict[str, int]): Training and evaluation counters
    """

    def __init__(
        self,
        in_d: int,
        out_d: int,
        dist_type: str = "Simple",
        in_dist_d: Optional[int] = None,
        out_dist_d: Optional[int] = None,
        dim_per_component: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the MAD distance encoder.

        Args:
            in_d (int): Input dimension
            out_d (int): Output dimension
            dist_type (str): Type of distance metric ("WideNorm", "L1", "IQE", or "Simple")
            in_dist_d (Optional[int]): Input dimension for distance metric
            out_dist_d (Optional[int]): Output dimension for distance metric
            dim_per_component (Optional[int]): Dimensions per component for IQE
            hidden_dims (Optional[List[int]]): Hidden layer dimensions for encoder
        """
        super(MadDistEncoder, self).__init__()

        encoder_kwargs = {}
        if hidden_dims is not None:
            encoder_kwargs["hidden_dims"] = hidden_dims

        if dist_type == "Simple":
            self.encoder = Encoder(in_d, out_d, force_positive=False, **encoder_kwargs)
        else:
            self.encoder = Encoder(in_d, out_d, **encoder_kwargs)

        assert dist_type in ["WideNorm", "L1", "IQE", "Simple"], \
            "The distance type must be WideNorm or L1 or IQE or Simple"
        self.dist_type = dist_type
        
        if self.dist_type == "WideNorm":
            self.param_dist = torchqmet.WideNorm(out_d, in_dist_d, out_dist_d, symmetric=False)
        elif self.dist_type == "IQE":
            self.param_dist = torchqmet.IQE(out_d, dim_per_component=dim_per_component, reduction="maxmean")
        else:
            self.param_dist = None

        self.counters = {"train": 0, "eval": 0}

    def dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance between two latent space embeddings.

        Args:
            z1 (torch.Tensor): First latent space embedding
            z2 (torch.Tensor): Second latent space embedding

        Returns:
            torch.Tensor: Distance between the embeddings
        """
        if self.dist_type == "WideNorm":
            return self.param_dist(z1, z2)
        elif self.dist_type == "L1":
            return torch.norm((z1 - z2), p=1, dim=1)
        elif self.dist_type == "IQE":
            return self.param_dist(z1, z2)
        elif self.dist_type == "Simple":
            diffs = F.relu(z1 - z2)
            max_vals = diffs.max(dim=1).values
            mean_vals = diffs.mean(dim=1)
            alpha = 0.9
            return alpha * max_vals + (1 - alpha) * mean_vals

    def training_step(self, experience_replay: ErDist, optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single training step for the MAD distance model.

        Args:
            experience_replay (ErDist): Experience replay buffer
            optimizer (torch.optim.Optimizer): Optimizer for training
            config (Dict[str, Any]): Training configuration dictionary

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Objective loss
                - Constraint loss
                - Total loss
        """
        self.train()
        self.counters["train"] += 1

        def to_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(config["device"])

        def objective_loss(batch_size: int, d_thresh: float) -> torch.Tensor:
            """
            Compute the objective loss for distance prediction.
            """
            s1_o, s2_o, d_traj_o = experience_replay.get_batch(batch_size=batch_size, d_thresh=d_thresh)
            s1_o, s2_o, d_traj_o = to_device(s1_o), to_device(s2_o), to_device(d_traj_o)
            d_traj_o = d_traj_o * config["scaling_factor"]
            z1_o = self.encoder(s1_o)
            z2_o = self.encoder(s2_o)
            pred_d = self.dist(z1_o, z2_o)
            loss = ((((pred_d + 1e-8) / (d_traj_o + 1e-8)) - 1).pow(2)).mean()
            return loss

        def objective_loss_2(batch_size: int, max_dist_sep: float) -> torch.Tensor:
            """
            Compute the second objective loss for maximum distance separation.
            Returns zero if all sampled state pairs are identical.
            """
            s1_o, s2_o = experience_replay.get_states_batch(batch_size=batch_size)
            s1_o, s2_o = to_device(s1_o), to_device(s2_o)
            max_dist_sep = max_dist_sep * config["scaling_factor"]

            z1_o = self.encoder(s1_o)
            z2_o = self.encoder(s2_o)

            pred_d = self.dist(z1_o, z2_o)

            if (pred_d == 0).all():
                return torch.tensor(0.0, device=pred_d.device)

            loss = (F.relu(1 - pred_d / max_dist_sep)).pow(2).mean()
            return loss

        def constrain_loss(batch_size, d_thresh):
            if config["prioritization"]:
                s1_c, s2_c, d_traj_c, indexes = experience_replay.get_prioritized_batch_c(batch_size=batch_size)
            else:
                s1_c, s2_c, d_traj_c = experience_replay.get_batch(batch_size=batch_size, d_thresh=d_thresh)
            d_traj_c = d_traj_c * config["scaling_factor"]
            s1_c, s2_c, d_traj_c = to_device(s1_c), to_device(s2_c), to_device(d_traj_c)
            z1_c = self.encoder(s1_c)
            z2_c = self.encoder(s2_c)
            pred_dist = self.dist(z1_c, z2_c)
            violation = F.relu(pred_dist - d_traj_c)
            if config["prioritization"]:
                experience_replay.update_priorities(indexes, violation.detach())
            return (violation ** 2).sum()

        loss_o = (config["weight_objective_1"] * objective_loss(config["batch_size_o"], config["max_dist_obj"]) +
                 config["weight_objective_2"] * objective_loss_2(config["batch_size_o"], config["max_dist_accuracy"]))

        if config["batch_size_c"] > 1024:
            loss_c = 0
            total_batch = config["batch_size_c"]
            while total_batch > 0:
                batch_size = min(total_batch, 1024)
                loss_c += constrain_loss(batch_size, config["max_dist_con"])
                total_batch = max(total_batch - 1024, 0)
        else:
            loss_c = constrain_loss(config["batch_size_c"], config["max_dist_con"])

        loss = loss_o + config["weight_constrains"] * loss_c

        # Only compute L1 regularization if lambda > 0
        l1_lambda = config.get("l1_lambda", 0.0)
        if l1_lambda > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += l1_lambda * l1_reg

        # Only compute L2 regularization if lambda > 0
        l2_lambda = config.get("l2_lambda", 0.0)
        if l2_lambda > 0:
            l2_reg = sum((param ** 2).sum() for param in self.parameters())
            loss += l2_lambda * l2_reg


        optimizer.zero_grad()
        loss.backward()
        if config["max_grad_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), config["max_grad_norm"])
        optimizer.step()

        return loss_o.cpu().detach(), loss_c.cpu().detach(), loss.cpu().detach()


class MadDistOrEncoder(torch.nn.Module):
    """
    Neural network model implementing the original MAD loss from the paper
    "State Representation Learning for Goal-Conditioned Reinforcement Learning"
    (Steccanella & Jonsson, 2022).

    Loss (equation 4 in the paper):
        L = (1/d_TD^2) * (d(φ(s), φ(s')) - d_TD)^2
          + (1/d_TD^2) * max(0, d(φ(s), φ(s')) - d_TD)^2

    Supports two distance types:
        "L1"     — symmetric L1 norm (paper default)
        "Simple" — asymmetric max/mean of ReLU differences (same as MadDistEncoder)
    """

    def __init__(self, in_d: int, out_d: int, dist_type: str = "L1") -> None:
        assert dist_type in ("L1", "Simple"), "dist_type must be 'L1' or 'Simple'"
        super(MadDistOrEncoder, self).__init__()
        self.dist_type = dist_type
        self.encoder = Encoder(in_d, out_d, force_positive=(dist_type == "Simple"))
        self.counters = {"train": 0, "eval": 0}

    def dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if self.dist_type == "L1":
            return torch.norm(z1 - z2, p=1, dim=1)
        else:  # Simple
            diffs = F.relu(z1 - z2)
            max_vals = diffs.max(dim=1).values
            mean_vals = diffs.mean(dim=1)
            alpha = 0.9
            return alpha * max_vals + (1 - alpha) * mean_vals

    def training_step(
        self,
        experience_replay: ErDist,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.train()
        self.counters["train"] += 1

        def to_device(t: torch.Tensor) -> torch.Tensor:
            return t.to(config["device"])

        s1, s2, d_traj = experience_replay.get_batch(
            batch_size=config["batch_size_o"],
            d_thresh=config.get("max_dist_traj_batch", None),
        )
        s1, s2, d_traj = to_device(s1), to_device(s2), to_device(d_traj)
        d_traj = d_traj * config["scaling_factor"]

        z1 = self.encoder(s1)
        z2 = self.encoder(s2)
        pred_d = self.dist(z1, z2)

        # Per-sample weight: 1 / d_TD^2  (equation 4)
        weight = 1.0 / (d_traj ** 2 + 1e-8)

        # Objective: weighted squared error
        loss_o = (weight * (pred_d - d_traj).pow(2)).mean()

        # Constraint penalty: penalise pred_d > d_traj (upper-bound violation)
        loss_c = (weight * F.relu(pred_d - d_traj).pow(2)).mean()

        loss = loss_o + config["weight_constrains"] *loss_c

        optimizer.zero_grad()
        loss.backward()
        if config.get("max_grad_norm") is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), config["max_grad_norm"])
        optimizer.step()

        return loss_o.cpu().detach(), loss_c.cpu().detach(), loss.cpu().detach()


class TDMadDistEncoder(torch.nn.Module):
    """
    Neural network model for learning the Temporal Difference MAD (TD-MAD) embedding.
    
    This model extends the MAD distance by incorporating temporal difference learning
    to better capture temporal relationships between states.
    
    Attributes:
        encoder (Encoder): Neural network encoder
        target_encoder (Encoder): Target network for temporal difference learning
        dist_type (str): Type of distance metric to use
        param_dist (Optional[torch.nn.Module]): Parameterized distance function
        counters (Dict[str, int]): Training and evaluation counters
    """

    def __init__(
        self,
        in_d: int,
        out_d: int,
        dist_type: str = "Simple",
        in_dist_d: Optional[int] = None,
        out_dist_d: Optional[int] = None,
        dim_per_component: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the TD-MAD distance encoder.

        Args:
            in_d (int): Input dimension
            out_d (int): Output dimension
            dist_type (str): Type of distance metric ("WideNorm", "L1", "IQE", or "Simple")
            in_dist_d (Optional[int]): Input dimension for distance metric
            out_dist_d (Optional[int]): Output dimension for distance metric
            dim_per_component (Optional[int]): Dimensions per component for IQE
            hidden_dims (Optional[List[int]]): Hidden layer dimensions for encoder
        """
        super(TDMadDistEncoder, self).__init__()

        encoder_kwargs = {}
        if hidden_dims is not None:
            encoder_kwargs["hidden_dims"] = hidden_dims

        if dist_type == "Simple":
            self.encoder = Encoder(in_d, out_d, force_positive=False, **encoder_kwargs)
        else:
            self.encoder = Encoder(in_d, out_d, **encoder_kwargs)

        assert dist_type in ["WideNorm", "L1", "IQE", "Simple"], \
            "The distance type must be WideNorm or L1 or IQE or Simple"
        self.dist_type = dist_type
        
        if self.dist_type == "WideNorm":
            self.param_dist = torchqmet.WideNorm(out_d, in_dist_d, out_dist_d, symmetric=False)
        elif self.dist_type == "IQE":
            self.param_dist = torchqmet.IQE(out_d, dim_per_component=dim_per_component, reduction="maxmean")
        else:
            self.param_dist = None

        self.counters = {"train": 0, "eval": 0}

        # Create and initialize target encoder
        self.target_encoder = Encoder(in_d, out_d)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_encoder.eval()

    def dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance between two latent space embeddings.

        Args:
            z1 (torch.Tensor): First latent space embedding
            z2 (torch.Tensor): Second latent space embedding

        Returns:
            torch.Tensor: Distance between the embeddings
        """
        if self.dist_type == "WideNorm":
            return self.param_dist(z1, z2)
        elif self.dist_type == "L1":
            return torch.norm((z1 - z2), p=1, dim=1)
        elif self.dist_type == "IQE":
            return self.param_dist(z1, z2)
        elif self.dist_type == "Simple":
            diffs = F.relu(z1 - z2)
            max_vals = diffs.max(dim=1).values
            mean_vals = diffs.mean(dim=1)
            alpha = 0.9
            return alpha * max_vals + (1 - alpha) * mean_vals

    def training_step(self, experience_replay: ErDist, optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single training step for the TD-MAD distance model.

        Args:
            experience_replay (ErDist): Experience replay buffer
            optimizer (torch.optim.Optimizer): Optimizer for training
            config (Dict[str, Any]): Training configuration dictionary

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Objective loss
                - Constraint loss
                - Total loss
        """
        self.train()
        self.counters["train"] += 1

        def to_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(config["device"])

        def update_target_network(model1: nn.Module, model2: nn.Module, tau: float = 0.005) -> None:
            """Soft update of target network parameters."""
            for param, target_param in zip(model1.parameters(), model2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        def objective_loss(batch_size: int, max_dist_sep: float) -> torch.Tensor:
            """Compute the temporal difference objective loss."""
            s_o, s__o, g_o, d_traj_o = experience_replay.get_triplet_batch(batch_size=batch_size, d_thresh=max_dist_sep)
            rand_goals = experience_replay.get_state_batch(batch_size=batch_size)
            s_o, s__o, g_o, d_traj_o, rand_goals = to_device(s_o), to_device(s__o), to_device(g_o), to_device(d_traj_o), to_device(rand_goals)

            z = self.encoder(s_o)
            z_goal = self.encoder(g_o)
            z_rand_goal = self.encoder(rand_goals)

            with torch.no_grad():
                z_next_target = self.target_encoder(s__o)
                z_goal_target = self.target_encoder(g_o)
                target = 1 + self.dist(z_next_target, z_goal_target)
                target = torch.min(target, d_traj_o).detach()

                z_rand_goal_target = self.target_encoder(rand_goals)
                target_rand = 1 + self.dist(z_next_target, z_rand_goal_target)

            loss = ((self.dist(z, z_goal) / (target + 1e-4)) - 1).pow(2).mean()
            loss_rand = ((self.dist(z, z_rand_goal) / (target_rand + 1e-4)) - 1).pow(2).mean()

            loss = loss + config["weight_objective_2"] * loss_rand

            update_target_network(self.encoder, self.target_encoder, tau=config["tau"])

            return loss + loss_rand

        def constrain_loss(batch_size: int, d_thresh: float) -> torch.Tensor:
            """Compute the constraint loss to ensure distance bounds."""
            if config["prioritization"]:
                s1_c, s2_c, d_traj_c, indexes = experience_replay.get_prioritized_batch_c(batch_size=batch_size)
            else:
                s1_c, s2_c, d_traj_c = experience_replay.get_batch(batch_size=batch_size, d_thresh=d_thresh)
            s1_c, s2_c, d_traj_c = to_device(s1_c), to_device(s2_c), to_device(d_traj_c)
            z1_c = self.encoder(s1_c)
            z2_c = self.encoder(s2_c)
            pred_dist = self.dist(z1_c, z2_c)
            violation = F.relu(pred_dist - d_traj_c)
            if config["prioritization"]:
                experience_replay.update_priorities(indexes, violation.detach())
            return (violation ** 2).sum()

        loss_o = objective_loss(config["batch_size_o"], config["max_dist_obj"])

        if config["batch_size_c"] > 512:
            loss_c = 0
            total_batch = config["batch_size_c"]
            while total_batch > 0:
                batch_size = min(total_batch, 512)
                loss_c += constrain_loss(batch_size, config["max_dist_con"])
                total_batch = max(total_batch - 512, 0)
        else:
            loss_c = constrain_loss(config["batch_size_c"], config["max_dist_con"])

        loss = loss_o + config["weight_constrains"] * loss_c

        # Only compute L1 regularization if lambda > 0
        l1_lambda = config.get("l1_lambda", 0.0)
        if l1_lambda > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += l1_lambda * l1_reg

        # Only compute L2 regularization if lambda > 0
        l2_lambda = config.get("l2_lambda", 0.0)
        if l2_lambda > 0:
            l2_reg = sum((param ** 2).sum() for param in self.parameters())
            loss += l2_lambda * l2_reg


        optimizer.zero_grad()
        loss.backward()
        if config["max_grad_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), config["max_grad_norm"])
        optimizer.step()

        return loss_o.cpu().detach(), loss_c.cpu().detach(), loss.cpu().detach()


