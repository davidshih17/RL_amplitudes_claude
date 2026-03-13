"""
Gymnasium environment for spinor-helicity amplitude simplification.

State representation:
- Per-bracket features for numerator brackets
- Global features (n_terms, n_brackets, n_point, etc.)

Action space:
- For reduced identities: 3 identity types × (bracket selection + parameters)
- Simplified: identity_idx × bracket_idx × param encodings

This environment is designed for SFT training where we have oracle trajectories.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import sympy as sp
from typing import Optional, List, Dict, Any, Tuple
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

from environment.bracket_env import ab, sb
from environment.utils import reorder_expr


# Identity types
IDENTITY_SCHOUTEN = 0
IDENTITY_MOMENTUM = 1
IDENTITY_MOMENTUM_SQ = 2
NUM_IDENTITIES = 3


def extract_bracket_features(bracket: sp.Function, power: int = 1, n_point: int = 6) -> np.ndarray:
    """
    Extract features from a single bracket.

    Features (total: 2 + 2*n_point = 14 for n_point=6):
    - bracket_type: 0 for angle, 1 for square (1 dim)
    - power: the exponent (log scale) (1 dim)
    - arg1_onehot: one-hot encoding of first argument (n_point dims)
    - arg2_onehot: one-hot encoding of second argument (n_point dims)
    """
    features = []

    # Bracket type (angle=0, square=1)
    bk_type = 0.0 if bracket.func.__name__ == 'ab' else 1.0
    features.append(bk_type)

    # Power (log scale, clipped)
    log_power = np.sign(power) * np.log1p(abs(power))
    features.append(np.clip(log_power, -5, 5))

    # Arguments as one-hot
    arg1, arg2 = bracket.args
    arg1_onehot = np.zeros(n_point)
    arg2_onehot = np.zeros(n_point)
    if 1 <= arg1 <= n_point:
        arg1_onehot[int(arg1) - 1] = 1.0
    if 1 <= arg2 <= n_point:
        arg2_onehot[int(arg2) - 1] = 1.0

    features.extend(arg1_onehot)
    features.extend(arg2_onehot)

    return np.array(features, dtype=np.float32)


def extract_term_features(term: sp.Expr, n_point: int = 6, max_brackets_per_term: int = 8) -> np.ndarray:
    """
    Extract features from a single numerator term.

    A term is a product of brackets with coefficients.
    Features:
    - coefficient_magnitude (log scale)
    - coefficient_sign
    - per-bracket features (padded to max_brackets_per_term)

    Total: 2 + max_brackets_per_term * bracket_feature_dim
    """
    bracket_feature_dim = 2 + 2 * n_point  # 14 for n_point=6

    features = []

    # Extract coefficient
    if term == 0:
        features.append(0.0)
        features.append(0.0)
    else:
        # Get coefficient (product of non-bracket factors)
        if isinstance(term, sp.Mul):
            coeff = 1
            for arg in term.args:
                if not isinstance(arg, (sp.Function, sp.Pow)):
                    coeff *= arg
                elif isinstance(arg, sp.Pow) and not isinstance(arg.base, sp.Function):
                    coeff *= arg
        elif isinstance(term, sp.Function):
            coeff = 1
        elif isinstance(term, sp.Pow) and isinstance(term.base, sp.Function):
            coeff = 1
        else:
            coeff = term

        try:
            coeff_float = float(coeff)
        except:
            coeff_float = 1.0

        log_mag = np.sign(coeff_float) * np.log1p(abs(coeff_float))
        features.append(np.clip(log_mag, -10, 10))
        features.append(1.0 if coeff_float >= 0 else -1.0)

    # Extract brackets with their powers
    bracket_infos = []

    def process_expr(expr, power_mult=1):
        if isinstance(expr, sp.Function) and expr.func.__name__ in ['ab', 'sb']:
            bracket_infos.append((expr, power_mult))
        elif isinstance(expr, sp.Pow):
            base, exp = expr.args
            if isinstance(base, sp.Function) and base.func.__name__ in ['ab', 'sb']:
                try:
                    bracket_infos.append((base, int(exp) * power_mult))
                except:
                    bracket_infos.append((base, power_mult))
            else:
                process_expr(base, power_mult)
        elif isinstance(expr, sp.Mul):
            for arg in expr.args:
                process_expr(arg, power_mult)

    if term != 0:
        process_expr(term)

    # Pad or truncate to max_brackets_per_term
    bracket_features = []
    for i in range(max_brackets_per_term):
        if i < len(bracket_infos):
            bk, power = bracket_infos[i]
            bracket_features.extend(extract_bracket_features(bk, power, n_point))
        else:
            # Padding with zeros
            bracket_features.extend([0.0] * bracket_feature_dim)

    features.extend(bracket_features)

    # Add validity flag
    features.append(1.0 if len(bracket_infos) > 0 else 0.0)

    return np.array(features, dtype=np.float32)


def extract_expression_features(
    sp_expr: sp.Expr,
    n_point: int = 6,
    max_terms: int = 20,
    max_brackets_per_term: int = 8,
) -> np.ndarray:
    """
    Extract features from a full spinor-helicity expression.

    Returns:
    - Per-term features (padded to max_terms)
    - Global features
    """
    if sp_expr == 0:
        num_terms = []
        denom = 1
    else:
        # Get numerator terms
        num, denom = sp.fraction(sp.cancel(sp_expr))
        if isinstance(num, sp.Add):
            num_terms = list(num.args)
        elif num == 0:
            num_terms = []
        else:
            num_terms = [num]

    # Per-term features
    bracket_feature_dim = 2 + 2 * n_point
    term_feature_dim = 2 + max_brackets_per_term * bracket_feature_dim + 1

    term_features = np.zeros((max_terms, term_feature_dim), dtype=np.float32)
    for i, term in enumerate(num_terms[:max_terms]):
        term_features[i] = extract_term_features(term, n_point, max_brackets_per_term)

    # Global features
    global_features = np.array([
        float(len(num_terms)),  # number of numerator terms
        float(n_point),  # number of external momenta
    ], dtype=np.float32)

    return term_features.flatten(), global_features


class SpinHelicityEnv(gym.Env):
    """
    Gymnasium environment for spinor-helicity simplification.

    State: Flattened term features + global features
    Action: Composite action encoding (identity, bracket_idx, params)

    For SFT, we primarily use this for feature extraction and action masking.
    """

    def __init__(
        self,
        n_point: int = 6,
        max_terms: int = 20,
        max_brackets_per_term: int = 8,
        max_steps: int = 50,
        dataset_path: Optional[str] = None,
    ):
        super().__init__()

        self.n_point = n_point
        self.max_terms = max_terms
        self.max_brackets_per_term = max_brackets_per_term
        self.max_steps = max_steps

        # Feature dimensions
        self.bracket_feature_dim = 2 + 2 * n_point
        self.term_feature_dim = 2 + max_brackets_per_term * self.bracket_feature_dim + 1
        self.global_feature_dim = 2

        # Observation space
        obs_dim = max_terms * self.term_feature_dim + self.global_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space for SFT
        # Simplified: we treat each (identity, bracket_idx, param_combo) as a discrete action
        # For Schouten on n_point momenta: need to choose pk, pl from remaining momenta
        # For Momentum: need to choose pout2
        # For Momentum_sq: need to choose mom_add subset

        # For now, use a simpler action space:
        # action = identity_idx * max_brackets + bracket_idx
        # Parameters will be inferred/stored separately

        self.max_brackets = max_terms * max_brackets_per_term  # Upper bound
        self.n_actions = NUM_IDENTITIES * self.max_brackets
        self.action_space = spaces.Discrete(self.n_actions)

        # Load dataset if provided
        if dataset_path is not None:
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = None

        # State
        self.func_dict = {'ab': ab, 'sb': sb}
        self.current_expr = None
        self.simple_expr = None
        self.current_step = 0
        self.brackets_list = []  # Current list of brackets

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and 'expression' in options:
            self.current_expr = options['expression']
            self.simple_expr = options.get('simple_expression', None)
        elif self.dataset is not None:
            idx = np.random.randint(len(self.dataset))
            sample = self.dataset[idx]
            self.current_expr = sample['scrambled_expression']
            self.simple_expr = sample['simple_expression']
        else:
            raise ValueError("No expression provided and no dataset loaded")

        self.current_step = 0
        self._update_brackets_list()

        obs = self._get_observation()
        info = {
            'num_terms': self._num_terms(),
            'action_mask': self._get_action_mask(),
        }

        return obs, info

    def _update_brackets_list(self):
        """Update the list of brackets in the current expression."""
        if self.current_expr == 0:
            self.brackets_list = []
            return

        num, _ = sp.fraction(self.current_expr)
        self.brackets_list = list(num.atoms(sp.Function))

    def _num_terms(self) -> int:
        """Count numerator terms."""
        if self.current_expr == 0:
            return 0
        num, _ = sp.fraction(self.current_expr)
        if isinstance(num, sp.Add):
            return len(num.args)
        elif num == 0:
            return 0
        return 1

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        term_features, global_features = extract_expression_features(
            self.current_expr,
            self.n_point,
            self.max_terms,
            self.max_brackets_per_term,
        )
        obs = np.concatenate([term_features, global_features])
        return np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

    def _get_action_mask(self) -> np.ndarray:
        """Get mask for valid actions."""
        mask = np.zeros(self.n_actions, dtype=np.float32)

        n_brackets = len(self.brackets_list)
        if n_brackets == 0:
            return mask

        # For each identity, allow actions on existing brackets
        for identity_idx in range(NUM_IDENTITIES):
            for bracket_idx in range(min(n_brackets, self.max_brackets)):
                action = identity_idx * self.max_brackets + bracket_idx
                if action < self.n_actions:
                    mask[action] = 1.0

        return mask

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action into (identity_idx, bracket_idx)."""
        identity_idx = action // self.max_brackets
        bracket_idx = action % self.max_brackets
        return identity_idx, bracket_idx

    def step(self, action: int):
        """
        Take a step in the environment.

        Note: For SFT training, we don't actually execute actions.
        This is mainly for evaluation purposes.
        """
        self.current_step += 1

        identity_idx, bracket_idx = self._decode_action(action)

        # Check validity
        if bracket_idx >= len(self.brackets_list):
            # Invalid action
            obs = self._get_observation()
            info = {
                'num_terms': self._num_terms(),
                'action_mask': self._get_action_mask(),
                'invalid_action': True,
            }
            return obs, -0.5, False, self.current_step >= self.max_steps, info

        # For now, we don't execute the action (would need parameter inference)
        # This is a placeholder for future RL training

        obs = self._get_observation()
        terminated = self._num_terms() == 0
        truncated = self.current_step >= self.max_steps

        info = {
            'num_terms': self._num_terms(),
            'action_mask': self._get_action_mask(),
        }

        reward = 0.0  # Placeholder

        return obs, reward, terminated, truncated, info


def build_observation_from_expr(
    sp_expr: sp.Expr,
    n_point: int = 6,
    max_terms: int = 20,
    max_brackets_per_term: int = 8,
) -> np.ndarray:
    """
    Utility function to build observation from a sympy expression.
    Used for SFT data preparation.
    """
    term_features, global_features = extract_expression_features(
        sp_expr, n_point, max_terms, max_brackets_per_term
    )
    obs = np.concatenate([term_features, global_features])
    return np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)


def get_bracket_index(bracket: sp.Function, brackets_list: List[sp.Function]) -> int:
    """
    Get the index of a bracket in the brackets list.
    Returns -1 if not found.
    """
    for i, bk in enumerate(brackets_list):
        if str(bk) == str(bracket):
            return i
    return -1


if __name__ == "__main__":
    # Test the environment
    print("Testing SpinHelicityEnv...")

    # Create a simple test expression
    test_expr = ab(1, 2) * sb(2, 3) / (ab(1, 3) * sb(1, 2))
    test_expr = reorder_expr(test_expr)

    # Test feature extraction
    print(f"\nTest expression: {test_expr}")

    term_features, global_features = extract_expression_features(test_expr, n_point=4)
    print(f"Term features shape: {term_features.shape}")
    print(f"Global features: {global_features}")

    # Test observation building
    obs = build_observation_from_expr(test_expr, n_point=4)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
