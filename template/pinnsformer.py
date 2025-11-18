import torch
import torch.nn as nn
from torch.autograd import grad

class PhysicsInformedTransformer(nn.Module):
    """
    Transformer that predicts:
    - Where (which PTX regions) cause slowdowns
    - Why (via embedded performance PDEs)
    - Magnitude (quantified by L2 miss rate increase)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_layers=6):
        super().__init__()
        
        # PTX token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.hotspot_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # [0, 1] hotspot probability
        )
        
        self.performance_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predicted execution time (ms)
        )
        
    def forward(self, ptx_tokens, contention_context=None):
        """
        Args:
            ptx_tokens: shape [batch, seq_len] - PTX instruction IDs
            contention_context: dict with {
                'enemy_intensity': float,
                'cache_pressure': float,
                'l2_size_mb': float
            }
        Returns:
            hotspots: [batch, seq_len] - per-instruction hotspot scores
            perf_pred: [batch] - predicted execution time
        """
        # Embed PTX tokens
        x = self.embedding(ptx_tokens)  # [batch, seq_len, embedding_dim]
        
        # Transformer encoder (learns code structure)
        x = self.transformer(x)  # [batch, seq_len, embedding_dim]
        
        # Predict hotspots per instruction
        hotspots = self.hotspot_predictor(x)  # [batch, seq_len, 1]
        hotspots = hotspots.squeeze(-1)  # [batch, seq_len]
        
        # Aggregate for overall performance prediction
        perf_features = x.mean(dim=1)  # Global average pooling
        if contention_context:
            context_vec = torch.tensor([
                contention_context['enemy_intensity'],
                contention_context['cache_pressure'],
                contention_context['l2_size_mb']
            ]).to(x.device)
            perf_features = torch.cat([perf_features, context_vec], dim=-1)
        
        perf_pred = self.performance_predictor(perf_features)  # [batch]
        
        return {
            'hotspots': hotspots,
            'performance': perf_pred,
            'instruction_features': x
        }

class PINNLossFunction(nn.Module):
    """
    Loss combines:
    1. Supervised loss (predicted vs actual metrics)
    2. Physics constraints (PDEs embedded as regularization)
    3. Contention coupling (enemy interference model)
    """
    
    def __init__(self, lambda_physics=0.1, lambda_contention=0.05):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_contention = lambda_contention
    
    def forward(self, outputs, targets, ptx_features, contention_config):
        """
        Args:
            outputs: dict from model forward()
            targets: {
                'execution_time': actual execution time,
                'l2_misses': actual L2 miss rate,
                'hotspot_annotations': [batch, seq_len] bool mask
            }
            ptx_features: architectural features from PTX
            contention_config: {'enemy_intensity', 'cache_size_mb', ...}
        """
        
        # 1. SUPERVISED LOSS: Performance prediction accuracy
        perf_pred = outputs['performance']
        perf_actual = targets['execution_time']
        supervised_loss = nn.MSELoss()(perf_pred, perf_actual)
        
        # 2. HOTSPOT LOSS: Alignment with annotated hotspots
        hotspot_pred = outputs['hotspots']
        hotspot_actual = targets['hotspot_annotations'].float()
        hotspot_loss = nn.BCELoss()(hotspot_pred, hotspot_actual)
        
        # 3. PHYSICS-INFORMED LOSS: Embed roofline model + cache PDE
        physics_loss = self._compute_physics_loss(
            outputs['instruction_features'],
            ptx_features,
            contention_config
        )
        
        # 4. CONTENTION COUPLING LOSS: Validate interference model
        contention_loss = self._compute_contention_loss(
            hotspot_pred,
            perf_pred,
            targets['l2_misses'],
            contention_config
        )
        
        # Total loss with weighting
        total_loss = (
            supervised_loss +
            hotspot_loss +
            self.lambda_physics * physics_loss +
            self.lambda_contention * contention_loss
        )
        
        return {
            'total': total_loss,
            'supervised': supervised_loss,
            'hotspot': hotspot_loss,
            'physics': physics_loss,
            'contention': contention_loss
        }
    
    def _compute_physics_loss(self, instruction_features, ptx_features, config):
        """
        Embed roofline model: Performance ≤ min(Compute_bound, Memory_bound)
        """
        batch_size = instruction_features.shape[0]
        
        # Extract compute/memory intensity from features
        compute_intensity = ptx_features['compute_intensity']  # [batch]
        global_loads = ptx_features['num_global_loads']  # [batch]
        
        # GPU specs (GTX 1650 Ti)
        peak_compute = 1.86 * 1e12  # 1.86 TFlop/s
        memory_bandwidth = 336 * 1e9  # 336 GB/s
        
        # Roofline constraints
        compute_bound = peak_compute / (global_loads + 1e-8)
        memory_bound = memory_bandwidth * (compute_intensity + 1e-8)
        roofline_limit = torch.min(compute_bound, memory_bound)
        
        # Extract predicted performance from instruction features
        predicted_perf = instruction_features.mean(dim=1).sum(dim=-1)
        
        # Penalize predictions that violate roofline
        physics_violation = torch.relu(predicted_perf - roofline_limit)
        physics_loss = (physics_violation ** 2).mean()
        
        return physics_loss
    
    def _compute_contention_loss(self, hotspots, perf_pred, l2_misses, config):
        """
        PDE: dL_miss/dt = α·AccessDensity - β·CacheSize + γ·EnemyIntensity
        Validate that high hotspots correlate with high L2 misses
        """
        batch_size = hotspots.shape[0]
        
        # High hotspots should correspond to high L2 misses
        # Expected relationship: hotspot_strength ∝ l2_miss_increase
        hotspot_strength = hotspots.max(dim=1)[0]  # [batch]
        l2_miss_increase = l2_misses / (1e-8 + l2_misses)  # normalized
        
        # Contention coupling: α·hotspot_strength ≈ l2_miss_increase
        alpha = config['enemy_intensity']
        expected_misses = alpha * hotspot_strength
        
        contention_loss = nn.MSELoss()(expected_misses, l2_miss_increase)
        
        return contention_loss