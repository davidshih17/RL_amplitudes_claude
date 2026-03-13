"""
SFT training with term-specific action space (4pt).

Uses TermSpecificActionSpace with 1452 actions:
- 12 brackets (for 4-point)
- 11 term_slots: 1 (whole expr) + 10 (individual terms)
- 11 identity types per bracket (1 Schouten + 6 Momentum + 4 Momentum_sq)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

from spinhelicity_env import build_observation_from_expr
from action_space_4pt import TermSpecificActionSpace
from train_sft_transformer import TransformerPolicySFT


@dataclass
class MultiLabelTransition:
    """Transition with multiple correct actions."""
    obs: np.ndarray
    action_mask: np.ndarray
    expert_action: int
    equivalent_actions: np.ndarray


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, transitions: List[MultiLabelTransition]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        equiv_indices = np.where(t.equivalent_actions > 0)[0]
        expert_action = int(equiv_indices.min()) if len(equiv_indices) > 0 else t.expert_action
        return {
            'obs': torch.tensor(t.obs, dtype=torch.float32),
            'action_mask': torch.tensor(t.action_mask, dtype=torch.float32),
            'expert_action': torch.tensor(expert_action, dtype=torch.long),
            'equivalent_actions': torch.tensor(t.equivalent_actions, dtype=torch.float32),
        }


def load_multilabel_transitions(path: str) -> List[MultiLabelTransition]:
    """Load pre-converted multi-label transitions."""
    print(f"Loading multi-label transitions from {path}...", flush=True)

    if os.path.isdir(path):
        all_transitions = []
        pkl_files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])
        print(f"  Found {len(pkl_files)} .pkl files in directory", flush=True)
        for fname in pkl_files:
            fpath = os.path.join(path, fname)
            with open(fpath, 'rb') as f:
                transitions = pickle.load(f)
                all_transitions.extend(transitions)
        print(f"  Loaded {len(all_transitions)} transitions total.", flush=True)
        return all_transitions
    else:
        with open(path, 'rb') as f:
            transitions = pickle.load(f)
        print(f"  Loaded {len(transitions)} transitions.", flush=True)
        return transitions


def train_sft_multilabel(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    output_dir: str,
    n_actions: int,
    warmup_epochs: int = 5,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
):
    """Train with multi-label loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(epoch):
        actual_epoch = epoch + start_epoch
        if actual_epoch < warmup_epochs:
            return (actual_epoch + 1) / warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    FLOAT_MIN = -3.4e38

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            obs = batch['obs'].to(device)
            action_mask = batch['action_mask'].to(device)
            equivalent_actions = batch['equivalent_actions'].to(device)

            optimizer.zero_grad()
            logits = model(obs)

            inf_mask = torch.clamp(torch.log(action_mask + 1e-10), min=FLOAT_MIN)
            masked_logits = logits + inf_mask

            # Soft label cross-entropy
            n_equiv = equivalent_actions.sum(dim=1, keepdim=True).clamp(min=1.0)
            soft_targets = equivalent_actions / n_equiv
            log_probs = F.log_softmax(masked_logits, dim=-1)
            loss = -(soft_targets * log_probs).sum(dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * obs.size(0)

            preds = masked_logits.argmax(dim=-1)
            batch_size = preds.size(0)
            for i in range(batch_size):
                if equivalent_actions[i, preds[i]] > 0:
                    train_correct += 1
            train_total += batch_size

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
                equivalent_actions = batch['equivalent_actions'].to(device)

                logits = model(obs)
                inf_mask = torch.clamp(torch.log(action_mask + 1e-10), min=FLOAT_MIN)
                masked_logits = logits + inf_mask

                n_equiv = equivalent_actions.sum(dim=1, keepdim=True).clamp(min=1.0)
                soft_targets = equivalent_actions / n_equiv
                log_probs = F.log_softmax(masked_logits, dim=-1)
                loss = -(soft_targets * log_probs).sum(dim=-1).mean()

                val_loss += loss.item() * obs.size(0)

                preds = masked_logits.argmax(dim=-1)
                batch_size = preds.size(0)
                for i in range(batch_size):
                    if equivalent_actions[i, preds[i]] > 0:
                        val_correct += 1
                val_total += batch_size

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

    return val_loss, best_val_acc


def train(args):
    """Main training function."""
    print("=" * 70)
    print("SFT Training with Term-Specific Action Space (4pt)")
    print(f"Started: {datetime.now()}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create action space
    action_space = TermSpecificActionSpace(
        n_point=args.n_point,
        max_terms=args.max_terms_action,
        max_brackets=args.max_brackets
    )
    print(f"\nAction space: {action_space.identities_per_bracket} identities/bracket, "
          f"{action_space.n_term_slots} term_slots (0=whole, 1-10=terms), {action_space.n_actions} total actions", flush=True)

    # Compute term_feature_dim
    bracket_feature_dim = 2 + 2 * args.n_point
    term_feature_dim = 2 + args.max_brackets_per_term * bracket_feature_dim + 1

    # Load multi-label transitions
    print("\nLoading multi-label transitions...", flush=True)

    if args.train_split is not None:
        all_transitions = load_multilabel_transitions(args.train_data)
        n_total = len(all_transitions)
        n_train = int(n_total * args.train_split)
        indices = list(range(n_total))
        random.shuffle(indices)
        train_transitions = [all_transitions[i] for i in indices[:n_train]]
        val_transitions = [all_transitions[i] for i in indices[n_train:]]
        print(f"Split {n_total} transitions: {len(train_transitions)} train, {len(val_transitions)} val", flush=True)
    else:
        train_transitions = load_multilabel_transitions(args.train_data)
        val_transitions = load_multilabel_transitions(args.val_data)
        print(f"Train: {len(train_transitions)}, Val: {len(val_transitions)}", flush=True)

    train_n_equiv = np.mean([t.equivalent_actions.sum() for t in train_transitions])
    print(f"Average equivalent actions per transition: {train_n_equiv:.2f}")

    train_dataset = MultiLabelDataset(train_transitions)
    val_dataset = MultiLabelDataset(val_transitions)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model with new action space size
    model = TransformerPolicySFT(
        max_terms=args.max_terms,
        term_feature_dim=term_feature_dim,
        n_actions=action_space.n_actions,
        max_brackets=args.max_brackets,
        actions_per_bracket=action_space.n_term_slots * action_space.identities_per_bracket,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        features_dim=args.features_dim,
        global_feature_dim=2,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Save config
    config = vars(args).copy()
    config['term_feature_dim'] = term_feature_dim
    config['n_actions'] = action_space.n_actions
    with open(os.path.join(args.output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nLoading weights from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if getattr(args, 'finetune', False):
            print("  -> Fine-tuning mode: starting from epoch 0")
        else:
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"  -> Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    print("\n" + "=" * 70)
    print("Starting SFT training with term-specific actions...")
    print("=" * 70)
    start_time = time.time()

    best_loss, best_acc = train_sft_multilabel(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        n_actions=action_space.n_actions,
        warmup_epochs=args.warmup_epochs,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='SFT training with term-specific action space')

    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--n_point', type=int, default=4)
    parser.add_argument('--max_terms', type=int, default=20)
    parser.add_argument('--max_terms_action', type=int, default=10)
    parser.add_argument('--max_brackets_per_term', type=int, default=8)
    parser.add_argument('--max_brackets', type=int, default=12)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ff_dim', type=int, default=128)
    parser.add_argument('--features_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_split', type=float, default=None)

    args = parser.parse_args()

    if args.val_data is None and args.train_split is None:
        parser.error("Either --val_data or --train_split must be provided")

    train(args)


if __name__ == '__main__':
    main()
