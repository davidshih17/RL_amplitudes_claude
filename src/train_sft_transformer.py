"""
SFT training using oracle trajectories with TRANSFORMER architecture.
Adapted from RL_dilogs_claude for spinor-helicity simplification.

Key differences from dilogs:
- Action space: bracket_idx * actions_per_bracket + action_type_idx (600 actions for n_point=4)
- Observations: per-term features for spinor-helicity expressions

Usage:
    python train_sft_transformer.py --output_dir ../models/sft_transformer_v1
"""

import argparse
import os
import sys
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

from spinhelicity_env import build_observation_from_expr


# ============================================================================
# Transformer Architecture (adapted from RL_dilogs_claude)
# ============================================================================

def ortho_init(module: nn.Module, gain: float = 1.0):
    """Orthogonal initialization matching SB3's default."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for spinor-helicity terms.

    Each term attends to all other terms, allowing the model to learn
    which brackets might simplify together.
    """

    def __init__(
        self,
        max_terms: int = 20,
        term_feature_dim: int = 83,  # Will be computed from observation
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        dropout: float = 0.1,
        features_dim: int = 128,
        global_feature_dim: int = 2,
    ):
        super().__init__()

        self.max_terms = max_terms
        self.term_feature_dim = term_feature_dim
        self.global_feature_dim = global_feature_dim
        self.embed_dim = embed_dim

        # Term embedding: project raw features to embed_dim
        self.term_embedding = nn.Linear(term_feature_dim, embed_dim)

        # Learnable [CLS] token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim + global_feature_dim, features_dim),
            nn.ReLU(),
        )

        self.output_dim = features_dim
        self._init_weights()

    def _init_weights(self):
        sqrt2 = np.sqrt(2.0)
        ortho_init(self.term_embedding, gain=sqrt2)
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                ortho_init(layer, gain=sqrt2)

    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            observations: (batch, obs_dim)
                where obs_dim = max_terms * term_feature_dim + global_feature_dim

        Returns:
            Dict with 'features', 'term_features', 'term_mask'
        """
        batch_size = observations.shape[0]

        # Split observation: term features + global features
        term_obs_flat = observations[:, :-self.global_feature_dim]
        global_features = observations[:, -self.global_feature_dim:]

        # Reshape to (batch, max_terms, term_feature_dim)
        term_obs = term_obs_flat.view(batch_size, self.max_terms, self.term_feature_dim)

        # Validity mask: use first feature (coefficient) as indicator
        # Terms with zero coefficient are invalid
        term_mask = (term_obs[:, :, 0] != 0).float()  # (batch, max_terms)

        # Embed terms
        term_embeddings = self.term_embedding(term_obs)  # (batch, max_terms, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, term_embeddings], dim=1)

        # Attention mask
        cls_mask = torch.zeros(batch_size, 1, device=observations.device, dtype=torch.bool)
        term_attn_mask = (term_mask == 0)
        attn_mask = torch.cat([cls_mask, term_attn_mask], dim=1)

        # Apply transformer
        transformed = self.transformer(sequence, src_key_padding_mask=attn_mask)

        # Extract outputs
        cls_output = transformed[:, 0, :]
        term_outputs = transformed[:, 1:, :]
        term_outputs = term_outputs * term_mask.unsqueeze(-1)

        # Global features
        global_input = torch.cat([cls_output, global_features], dim=-1)
        features = self.output_projection(global_input)

        return {
            'features': features,
            'term_features': term_outputs,
            'term_mask': term_mask,
        }


class TransformerPolicySFT(nn.Module):
    """
    Transformer policy for SFT training on spinor-helicity.

    Action space: bracket_idx * actions_per_bracket + action_type_idx
    For n_point=4: 50 brackets × 12 action types = 600 actions
    """

    def __init__(
        self,
        max_terms: int = 20,
        term_feature_dim: int = 83,
        n_actions: int = 600,
        max_brackets: int = 50,
        actions_per_bracket: int = 12,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        dropout: float = 0.1,
        features_dim: int = 128,
        global_feature_dim: int = 2,
        pi_hidden: List[int] = None,
    ):
        super().__init__()

        if pi_hidden is None:
            pi_hidden = [64, 64]

        self.max_terms = max_terms
        self.max_brackets = max_brackets
        self.actions_per_bracket = actions_per_bracket
        self.n_actions = n_actions

        # Transformer encoder
        self.encoder = TransformerEncoder(
            max_terms=max_terms,
            term_feature_dim=term_feature_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            features_dim=features_dim,
            global_feature_dim=global_feature_dim,
        )

        # Policy head: global features -> action logits
        # Since brackets span across terms, we use global representation
        self.pi = nn.Sequential(
            nn.Linear(features_dim, pi_hidden[0]),
            nn.ReLU(),
            nn.Linear(pi_hidden[0], pi_hidden[1]),
            nn.ReLU(),
            nn.Linear(pi_hidden[1], n_actions),
        )

        self._init_policy_head()

    def _init_policy_head(self):
        sqrt2 = np.sqrt(2.0)
        pi_layers = list(self.pi)
        for i, layer in enumerate(pi_layers):
            if isinstance(layer, nn.Linear):
                if i == len(pi_layers) - 1:
                    ortho_init(layer, gain=0.01)
                else:
                    ortho_init(layer, gain=sqrt2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)

        Returns:
            logits: (batch, n_actions)
        """
        encoder_out = self.encoder(obs)
        features = encoder_out['features']
        logits = self.pi(features)
        return logits


# ============================================================================
# Data Loading and Processing
# ============================================================================

@dataclass
class OracleTransition:
    obs: np.ndarray
    action_mask: np.ndarray
    expert_action: int


class OracleDataset(torch.utils.data.Dataset):
    def __init__(self, transitions: List[OracleTransition]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            'obs': torch.tensor(t.obs, dtype=torch.float32),
            'action_mask': torch.tensor(t.action_mask, dtype=torch.float32),
            'expert_action': torch.tensor(t.expert_action, dtype=torch.long),
        }


def load_oracle_data(data_path: str) -> List[Dict]:
    """Load oracle trajectory data."""
    print(f"Loading oracle data from {data_path}", flush=True)
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples", flush=True)
    return samples


def save_transitions(transitions: List[OracleTransition], path: str):
    """Save converted transitions to pickle file."""
    print(f"Saving {len(transitions)} transitions to {path}...", flush=True)
    with open(path, 'wb') as f:
        pickle.dump(transitions, f)
    print(f"  Saved.", flush=True)


def load_transitions(path: str) -> List[OracleTransition]:
    """Load pre-converted transitions from pickle file."""
    print(f"Loading transitions from {path}...", flush=True)
    with open(path, 'rb') as f:
        transitions = pickle.load(f)
    print(f"  Loaded {len(transitions)} transitions.", flush=True)
    return transitions


def oracle_to_transitions(
    samples: List[Dict],
    action_space,
    n_point: int = 4,
    max_terms: int = 20,
    max_brackets_per_term: int = 8,
) -> Tuple[List[OracleTransition], int]:
    """Convert oracle samples to training transitions."""
    transitions = []

    # Compute observation dimensions
    bracket_feature_dim = 2 + 2 * n_point  # type one-hot (2) + args one-hot (2 * n_point)
    term_feature_dim = 2 + max_brackets_per_term * bracket_feature_dim + 1  # coeff(2) + brackets + valid flag
    global_feature_dim = 2  # n_terms, n_brackets
    obs_dim = max_terms * term_feature_dim + global_feature_dim

    print(f"Observation dimensions:", flush=True)
    print(f"  term_feature_dim: {term_feature_dim}", flush=True)
    print(f"  global_feature_dim: {global_feature_dim}", flush=True)
    print(f"  total obs_dim: {obs_dim}", flush=True)

    skipped = 0
    total_samples = len(samples)
    for sample_idx, sample in enumerate(samples):
        if (sample_idx + 1) % 5000 == 0:
            print(f"  Processing sample {sample_idx + 1}/{total_samples}...", flush=True)
        for step in sample['trajectory']:
            state_expr = step['state']
            action_info = step['action']
            action_idx = action_info['action_idx']

            # Build observation
            try:
                obs = build_observation_from_expr(
                    state_expr, n_point, max_terms, max_brackets_per_term
                )
            except Exception as e:
                skipped += 1
                continue

            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

            # Build action mask
            # Get brackets from expression to determine valid actions
            import sympy as sp
            if state_expr == 0:
                brackets = []
            else:
                num, _ = sp.fraction(state_expr)
                brackets = list(num.atoms(sp.Function))
            action_mask = action_space.get_action_mask(brackets)

            transitions.append(OracleTransition(
                obs=obs,
                action_mask=action_mask,
                expert_action=action_idx,
            ))

    print(f"Created {len(transitions)} transitions (skipped {skipped})", flush=True)
    return transitions, term_feature_dim


def train_sft(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    output_dir: str,
    warmup_epochs: int = 5,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
):
    """Train the model using supervised learning on oracle data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(epoch):
        actual_epoch = epoch + start_epoch
        if actual_epoch < warmup_epochs:
            return (actual_epoch + 1) / warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    FLOAT_MIN = -3.4e38

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            obs = batch['obs'].to(device)
            action_mask = batch['action_mask'].to(device)
            expert_actions = batch['expert_action'].to(device)

            optimizer.zero_grad()
            logits = model(obs)

            # Apply action mask
            inf_mask = torch.clamp(torch.log(action_mask + 1e-10), min=FLOAT_MIN)
            masked_logits = logits + inf_mask

            loss = F.cross_entropy(masked_logits, expert_actions)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            preds = masked_logits.argmax(dim=-1)
            train_correct += (preds == expert_actions).sum().item()
            train_total += expert_actions.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                action_mask = batch['action_mask'].to(device)
                expert_actions = batch['expert_action'].to(device)

                logits = model(obs)
                inf_mask = torch.clamp(torch.log(action_mask + 1e-10), min=FLOAT_MIN)
                masked_logits = logits + inf_mask

                loss = F.cross_entropy(masked_logits, expert_actions)
                val_loss += loss.item() * obs.size(0)
                preds = masked_logits.argmax(dim=-1)
                val_correct += (preds == expert_actions).sum().item()
                val_total += expert_actions.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"lr={current_lr:.6f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt'))

    return best_val_loss, best_val_acc


def train(args):
    """Main training function."""
    print("=" * 70)
    print("SFT Training with Transformer Architecture")
    print(f"Started: {datetime.now()}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Transformer: embed_dim={args.embed_dim}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}, ff_dim={args.ff_dim}")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create action space
    from action_space import ActionSpace
    action_space = ActionSpace(n_point=args.n_point, max_brackets=args.max_brackets)
    print(f"\nAction space: {action_space.actions_per_bracket} actions/bracket, "
          f"{action_space.n_actions} total actions", flush=True)

    # Compute term_feature_dim for model creation
    bracket_feature_dim = 2 + 2 * args.n_point
    term_feature_dim = 2 + args.max_brackets_per_term * bracket_feature_dim + 1

    # Load pre-converted transitions
    print("\nLoading transitions...", flush=True)
    train_transitions = load_transitions(args.train_data)
    val_transitions = load_transitions(args.val_data)

    print(f"Train: {len(train_transitions)}, Val: {len(val_transitions)}", flush=True)

    train_dataset = OracleDataset(train_transitions)
    val_dataset = OracleDataset(val_transitions)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Compute global feature dim
    global_feature_dim = 2

    # Create model
    model = TransformerPolicySFT(
        max_terms=args.max_terms,
        term_feature_dim=term_feature_dim,
        n_actions=action_space.n_actions,
        max_brackets=args.max_brackets,
        actions_per_bracket=action_space.actions_per_bracket,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        features_dim=args.features_dim,
        global_feature_dim=global_feature_dim,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Save config
    config = vars(args).copy()
    config['term_feature_dim'] = term_feature_dim
    config['n_actions'] = action_space.n_actions
    config['actions_per_bracket'] = action_space.actions_per_bracket
    with open(os.path.join(args.output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"  Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    print("\n" + "=" * 70)
    print("Starting SFT training...")
    print("=" * 70)
    start_time = time.time()

    best_loss, best_acc = train_sft(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        warmup_epochs=args.warmup_epochs,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'best_val_loss': best_loss,
        'best_val_acc': best_acc,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    print(f"\nFinal model saved to {args.output_dir}")
    print(f"Training completed: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description='SFT training with transformer for spinor-helicity')

    # Data paths
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Environment config
    parser.add_argument('--n_point', type=int, default=4)
    parser.add_argument('--max_terms', type=int, default=20)
    parser.add_argument('--max_brackets_per_term', type=int, default=8)
    parser.add_argument('--max_brackets', type=int, default=50)

    # Training config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Transformer architecture
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ff_dim', type=int, default=128)
    parser.add_argument('--features_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Resume
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
