"""Shared neural training steps and explicit CNN1D/LSTM architectures."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from src.models import postprocess


def check_device(device):
    if device not in ('cpu', 'mps'):
        raise ValueError('training device must be cpu or mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable: use an Apple GPU with MPS-enabled PyTorch, or --device cpu')


def fit_networks(model, features, targets, validation=None, epochs=None):
    device = getattr(model, 'device', 'cpu')
    check_device(device)
    torch.manual_seed(model.seed)
    torch.set_num_threads(1)
    output_width = 4 if model.mode == 'softmax' else 3
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] not in (3, 4) or (output_width == 4 and targets.shape[1] != 4):
        raise ValueError('softmax requires four targets; joint/separate require three or four')
    targets = targets[:, :output_width]
    model.scaler_ = StandardScaler().fit(features)
    train_features = model.prepare_tensor(torch.tensor(model.scaler_.transform(features), dtype=torch.float32))
    train_targets = torch.tensor(targets, dtype=torch.float32)
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    if validation is not None:
        validation_features = model.prepare_tensor(torch.tensor(
            model.scaler_.transform(validation[0]), dtype=torch.float32
        ))
        validation_targets = np.asarray(validation[1])
        if validation_targets.ndim != 2 or validation_targets.shape[1] < output_width:
            raise ValueError('validation targets do not match the output mode')
        validation_targets = validation_targets[:, :output_width]
        validation_features = validation_features.to(device)

    if model.mode in ('joint', 'softmax'):
        networks = [model.build_network(features.shape[1], output_width)]
    else:
        networks = []
        for _ in range(3):
            networks.append(model.build_network(features.shape[1], 1))
    model.networks_ = nn.ModuleList(networks).to(device)

    optimizer = torch.optim.AdamW(
        model.networks_.parameters(),
        lr=model.params.get('lr', 1e-3),
        weight_decay=model.params.get('weight_decay', 1e-5),
    )
    batch_size = model.params.get('batch_size', 128)
    best_score = np.inf
    epochs_without_improvement = 0
    best_state = None
    model.best_epoch_ = 0
    max_epochs = epochs if epochs is not None else model.cfg.mlp_max_epochs
    for epoch in range(1, max_epochs + 1):
        model.epochs_trained_ = epoch
        model.networks_.train()
        # Keep the CPU shuffle and initialization shared by both training devices.
        order = torch.randperm(len(train_features)).to(device)
        for start in range(0, len(train_features), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            predictions = torch.cat(
                [network(train_features[indices]) for network in model.networks_],
                dim=1,
            )
            loss = nn.functional.mse_loss(predictions, train_targets[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            loss.backward()
            if model.gradient_clip is not None:
                # Each independent network has its own gradient-norm limit.
                for network in model.networks_:
                    norm = nn.utils.clip_grad_norm_(network.parameters(), model.gradient_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('non-finite gradient norm')
            optimizer.step()
        if validation is None:
            model.best_epoch_ = epoch
        if validation is not None:
            model.networks_.eval()
            with torch.no_grad():
                raw = torch.cat(
                    [network(validation_features) for network in model.networks_],
                    dim=1,
                ).cpu().numpy()
            if not np.isfinite(raw).all():
                raise FloatingPointError('non-finite early-stopping predictions')
            score = 100 * np.mean(
                np.abs(postprocess(raw) - postprocess(validation_targets))
            )
            if not np.isfinite(score):
                raise FloatingPointError('non-finite early-stopping score')
            if score < best_score - model.cfg.min_improvement_pp:
                best_score = score
                epochs_without_improvement = 0
                model.best_epoch_ = epoch
                best_state = copy.deepcopy(model.networks_.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= model.cfg.patience:
                    break
    if best_state is not None:
        model.networks_.load_state_dict(best_state)
    # Saved models and prediction work on machines without an Apple GPU too.
    model.networks_.cpu()
    model.n_parameters_ = sum(parameter.numel() for parameter in model.networks_.parameters())
    return model



def predict_networks(model, features):
    model.networks_.eval()
    features = torch.tensor(model.scaler_.transform(features), dtype=torch.float32)
    features = model.prepare_tensor(features)
    with torch.no_grad():
        predictions = torch.cat([network(features) for network in model.networks_], dim=1)
    return predictions.numpy()


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, output_width, use_softmax=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_width)
        self.use_softmax = use_softmax

    def forward(self, features):
        # No incoming state: each curve starts with zero hidden/cell state.
        _, (hidden, _) = self.lstm(features)
        output = self.output(hidden[-1])
        return torch.softmax(output, dim=1) if self.use_softmax else output


class TorchSequence:
    def __init__(self, kind, params, mode, seed, cfg, device='cpu'):
        if mode not in ('joint', 'separate', 'softmax'):
            raise ValueError('unknown output mode')
        self.params = params
        self.mode = mode
        self.seed = seed
        self.cfg = cfg
        self.device = device
        self.kind = kind
        self.gradient_clip = params.get('gradient_clip', 1.) if kind == 'lstm' else None

    def fit(self, features, targets, validation=None, epochs=None):
        return fit_networks(self, features, targets, validation, epochs)

    def predict(self, features):
        return predict_networks(self, features)

    def prepare_tensor(self, features):
        if self.kind == 'cnn1d':
            return features.unsqueeze(1)  # N, channels=1, K
        return features.unsqueeze(2)  # N, K, features=1

    def build_network(self, input_width, output_width):
        if self.kind == 'lstm':
            return LSTMNetwork(self.params.get('hidden_size', 32),
                               self.params.get('num_layers', 1), output_width,
                               use_softmax=self.mode == 'softmax')
        kernel_size = self.params.get('kernel_size', 3)
        channels = self.params.get('channels', (16, 32))
        head_width = self.params.get('head_width', 32)
        layers = []
        input_channels = 1
        for output_channels in channels:
            layers.append(nn.Conv1d(input_channels, output_channels,
                                    kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            input_channels = output_channels
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_channels * input_width, head_width))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(head_width, output_width))
        if self.mode == 'softmax':
            layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)
