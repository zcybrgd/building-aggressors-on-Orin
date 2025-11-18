import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class GPUPerformanceDataset(Dataset):
    """Load collected benchmark + profiling data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = self._load_samples()
        self.tokenizer = PTXTokenizer()
    
    def _load_samples(self):
        samples = []
        for json_file in self.data_dir.glob('*.json'):
            with open(json_file) as f:
                sample = json.load(f)
                samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize PTX
        ptx_tokens = self.tokenizer.encode_tokens(
            self.tokenizer.tokenize_ptx(sample['ptx_code'])
        )
        ptx_tokens = torch.tensor(ptx_tokens, dtype=torch.long)
        
        # Extract features
        ptx_features = extract_ptx_features(sample['ptx_code'])
        
        # Load ground truth metrics
        targets = {
            'execution_time': torch.tensor(sample['timing_ms'], dtype=torch.float32),
            'l2_misses': torch.tensor(sample['l2_misses'], dtype=torch.float32),
            'hotspot_annotations': torch.tensor(
                sample.get('hotspot_mask', [0] * len(ptx_tokens)),
                dtype=torch.bool
            )
        }
        
        # Contention context
        contention = {
            'enemy_intensity': sample.get('enemy_intensity', 0.0),
            'cache_pressure': sample.get('cache_pressure', 0.0),
            'l2_size_mb': 4.0  # GTX 1650 Ti specs
        }
        
        return {
            'ptx_tokens': ptx_tokens,
            'ptx_features': ptx_features,
            'targets': targets,
            'contention': contention
        }

class GPUPerformanceModule(pl.LightningModule):
    """PyTorch Lightning wrapper for training"""
    
    def __init__(self, vocab_size, lr=1e-3):
        super().__init__()
        self.model = PhysicsInformedTransformer(vocab_size)
        self.loss_fn = PINNLossFunction(lambda_physics=0.1, lambda_contention=0.05)
        self.lr = lr
    
    def forward(self, batch):
        ptx_tokens = batch['ptx_tokens']
        contention = batch['contention']
        return self.model(ptx_tokens, contention)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_dict = self.loss_fn(
            outputs,
            batch['targets'],
            batch['ptx_features'],
            batch['contention']
        )
        
        # Log losses
        for loss_name, loss_val in loss_dict.items():
            self.log(f'train_{loss_name}', loss_val)
        
        return loss_dict['total']
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_dict = self.loss_fn(
            outputs,
            batch['targets'],
            batch['ptx_features'],
            batch['contention']
        )
        
        for loss_name, loss_val in loss_dict.items():
            self.log(f'val_{loss_name}', loss_val)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Training
if __name__ == '__main__':
    dataset = GPUPerformanceDataset('./benchmark_data')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GPUPerformanceModule(vocab_size=5000)
    trainer = pl.Trainer(max_epochs=50, gpus=1)
    trainer.fit(model, train_loader)
