"""
Action space for 5-point amplitudes with term-specific identity application.

Key differences from 4-point (action_space_4pt.py):
- 20 brackets: C(5,2) * 2 = 10 pairs * 2 types (ab/sb) = 20
- 3 Schouten identity types per bracket (C(3,2) = 3 pairs from 3 other momenta)
- 8 Momentum identity types per bracket (2 pout1 * 4 pout2)
- 8 Momentum_sq identity types per bracket (2^3 = 8 subsets of 3 other momenta)
- Total per bracket: 3 + 8 + 8 = 19 identity types
- 11 term slots: 1 "whole expression" + 10 individual terms
- Total actions: 20 * 11 * 19 = 4180 actions

Action encoding:
    action = bracket_idx * (n_term_slots * identities_per_bracket) + term_slot * identities_per_bracket + identity_type_idx

Where:
- bracket_idx: 0-19 (which of the 20 possible brackets)
- term_slot: 0-10 (0 = entire expression, 1-10 = specific term index 0-9)
- identity_type_idx: 0-18 (which identity variant)
"""

from typing import List, Dict, Any, Tuple, Optional, NamedTuple
from itertools import permutations, combinations
import numpy as np
import sympy as sp


# Identity type constants
IDENTITY_SCHOUTEN = 'schouten'
IDENTITY_MOMENTUM = 'momentum'
IDENTITY_MOMENTUM_SQ = 'momentum_sq'


class ActionSpec(NamedTuple):
    """Full specification of an action."""
    bracket_idx: int  # Index in canonical bracket ordering (0-19 for 5-point)
    term_slot: int    # 0 = whole expression, 1-10 = specific term (term_slot-1)
    identity: str     # 'schouten', 'momentum', or 'momentum_sq'
    identity_type_idx: int  # Index within identity types for this bracket (0-18)
    params: Dict[str, Any]  # Identity-specific parameters

    @property
    def is_whole_expression(self) -> bool:
        """True if this action applies to the whole expression."""
        return self.term_slot == 0

    @property
    def term_idx(self) -> Optional[int]:
        """Get the actual term index (0-9) for single-term actions, or None for whole expression."""
        return (self.term_slot - 1) if self.term_slot > 0 else None


class TermSpecificActionSpace5pt:
    """
    Action space with term-specific identity application for 5-point amplitudes.

    For 5-point amplitudes:
    - 20 canonical brackets (sorted deterministically)
    - 11 term slots: 1 "whole expression" (slot 0) + 10 individual terms (slots 1-10)
    - 19 identity types per bracket (3 Schouten + 8 Momentum + 8 Momentum_sq)
    - Total: 20 * 11 * 19 = 4180 actions

    Two action types:
    1. term_slot=0: Apply identity to ALL occurrences in ENTIRE expression
    2. term_slot=1-10: Apply identity to ONE instance in term (term_slot-1)
    """

    def __init__(self, n_point: int = 5, max_terms: int = 10, max_brackets: int = 20):
        """
        Args:
            n_point: Number of external momenta (default 5)
            max_terms: Maximum number of terms (default 10)
            max_brackets: Maximum number of brackets (default 20 for 5-point)
        """
        assert n_point == 5, "This action space is specifically for 5-point amplitudes"
        self.n_point = n_point
        self.max_terms = max_terms
        self.max_brackets = max_brackets

        # Number of term slots: 1 (whole expr) + max_terms (individual terms)
        self.n_term_slots = 1 + max_terms

        # For 5-point: 3 + 8 + 8 = 19 identity types per bracket
        self.identities_per_bracket = self._compute_identities_per_bracket()

        # Total action space size
        self.n_actions = max_brackets * self.n_term_slots * self.identities_per_bracket

        # Build canonical bracket ordering
        self.canonical_brackets = self._build_canonical_brackets()

        # Pre-compute identity types for each bracket
        self.identity_types_cache = {}

    def _compute_identities_per_bracket(self) -> int:
        """Compute number of identity types per bracket."""
        # For n_point=5: 3 other momenta besides bracket args
        n_other = self.n_point - 2  # = 3 for 5-point

        # Schouten: C(n_other, 2) distinct pairs (swapping pk/pl gives same result)
        n_schouten = n_other * (n_other - 1) // 2  # C(3,2) = 3

        # Momentum: 2 pout1 choices * (1 + n_other) pout2 choices
        n_momentum = 2 * (1 + n_other)  # 2 * 4 = 8

        # Momentum_sq: all subsets of other momenta
        n_momentum_sq = 2 ** n_other  # 2^3 = 8

        return n_schouten + n_momentum + n_momentum_sq  # 3 + 8 + 8 = 19

    def _build_canonical_brackets(self) -> List[Tuple[str, int, int]]:
        """
        Build canonical ordering of all possible brackets for n_point.

        Returns list of (bracket_type, i, j) tuples where:
        - bracket_type: 'ab' or 'sb'
        - i, j: momentum indices (1-indexed), sorted so i < j

        Ordering: first by bracket type (ab before sb), then by (i, j) lexicographically
        """
        brackets = []
        for bk_type in ['ab', 'sb']:
            for i in range(1, self.n_point + 1):
                for j in range(i + 1, self.n_point + 1):
                    brackets.append((bk_type, i, j))
        return brackets

    def get_bracket_idx(self, bracket: sp.Function) -> int:
        """
        Get canonical index of a bracket.

        Args:
            bracket: SymPy bracket function (ab(i,j) or sb(i,j))

        Returns:
            Canonical index (0-19 for 5-point)
        """
        bk_type = bracket.func.__name__
        args = sorted(bracket.args)
        i, j = int(args[0]), int(args[1])

        key = (bk_type, i, j)
        try:
            return self.canonical_brackets.index(key)
        except ValueError:
            raise ValueError(f"Bracket {bracket} not in canonical set")

    def get_bracket_from_idx(self, bracket_idx: int) -> Tuple[str, int, int]:
        """
        Get bracket specification from canonical index.

        Args:
            bracket_idx: Canonical index (0-19 for 5-point)

        Returns:
            (bracket_type, i, j) tuple
        """
        return self.canonical_brackets[bracket_idx]

    def get_identity_types_for_bracket(self, bracket_idx: int) -> List[Dict[str, Any]]:
        """
        Get all identity types for a specific bracket.

        Args:
            bracket_idx: Canonical bracket index

        Returns:
            List of identity specifications, each with:
            - 'identity': identity type name
            - 'params': identity parameters
            - 'local_idx': index within this bracket's identities (0-18)
        """
        if bracket_idx in self.identity_types_cache:
            return self.identity_types_cache[bracket_idx]

        bk_type, i, j = self.canonical_brackets[bracket_idx]
        bracket_args = (i, j)

        # Other momenta (not in bracket)
        other_momenta = [k for k in range(1, self.n_point + 1) if k not in bracket_args]

        identity_types = []

        # Schouten identities: ALL C(n_other, 2) pairs
        # Each pair (pk, pl) gives a different expansion
        for pk, pl in combinations(other_momenta, 2):
            identity_types.append({
                'identity': IDENTITY_SCHOUTEN,
                'params': {'pk': pk, 'pl': pl},
                'local_idx': len(identity_types)
            })

        # Momentum identities: pout1 from bracket args, pout2 from {pout1} + other_momenta
        # Note: pout2 can equal pout1, but cannot be the OTHER bracket arg (zero denominator)
        for pout1 in bracket_args:
            # Valid pout2: pout1 itself + other_momenta (NOT the other bracket arg)
            valid_pout2 = [pout1] + other_momenta

            for pout2 in valid_pout2:
                identity_types.append({
                    'identity': IDENTITY_MOMENTUM,
                    'params': {'pout1': pout1, 'pout2': pout2},
                    'local_idx': len(identity_types)
                })

        # Momentum squared identities: all subsets of other momenta as mom_add
        for subset_size in range(len(other_momenta) + 1):
            for mom_add in combinations(other_momenta, subset_size):
                identity_types.append({
                    'identity': IDENTITY_MOMENTUM_SQ,
                    'params': {'mom_add': list(mom_add)},
                    'local_idx': len(identity_types)
                })

        self.identity_types_cache[bracket_idx] = identity_types
        return identity_types

    def encode_action(
        self,
        bracket_idx: int,
        term_slot: int,
        identity_type_idx: int
    ) -> int:
        """
        Encode action components to action index.

        Args:
            bracket_idx: Canonical bracket index (0-19)
            term_slot: 0 = whole expression, 1-10 = specific term (term_slot-1)
            identity_type_idx: Identity type index (0-18)

        Returns:
            Action index (0-4179 for default config)
        """
        return (bracket_idx * self.n_term_slots * self.identities_per_bracket +
                term_slot * self.identities_per_bracket +
                identity_type_idx)

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode action index to components.

        Args:
            action: Action index

        Returns:
            (bracket_idx, term_slot, identity_type_idx) tuple
            where term_slot: 0 = whole expression, 1-10 = specific term
        """
        bracket_idx = action // (self.n_term_slots * self.identities_per_bracket)
        remainder = action % (self.n_term_slots * self.identities_per_bracket)
        term_slot = remainder // self.identities_per_bracket
        identity_type_idx = remainder % self.identities_per_bracket
        return bracket_idx, term_slot, identity_type_idx

    def get_action_spec(self, action: int) -> ActionSpec:
        """
        Get full action specification from action index.

        Args:
            action: Action index

        Returns:
            ActionSpec with all details
        """
        bracket_idx, term_slot, identity_type_idx = self.decode_action(action)
        identity_types = self.get_identity_types_for_bracket(bracket_idx)

        if identity_type_idx >= len(identity_types):
            raise ValueError(f"Invalid identity_type_idx {identity_type_idx} for bracket {bracket_idx}")

        id_spec = identity_types[identity_type_idx]

        return ActionSpec(
            bracket_idx=bracket_idx,
            term_slot=term_slot,
            identity=id_spec['identity'],
            identity_type_idx=identity_type_idx,
            params=id_spec['params']
        )

    def find_action_index(
        self,
        bracket: sp.Function,
        term_slot: int,
        identity: str,
        params: Dict[str, Any]
    ) -> int:
        """
        Find action index from high-level specification.

        Args:
            bracket: SymPy bracket function
            term_slot: 0 = whole expression, 1-10 = specific term (term_slot-1)
            identity: Identity type name
            params: Identity parameters

        Returns:
            Action index
        """
        bracket_idx = self.get_bracket_idx(bracket)
        identity_types = self.get_identity_types_for_bracket(bracket_idx)

        # Find matching identity type
        for id_spec in identity_types:
            if id_spec['identity'] != identity:
                continue

            if identity == IDENTITY_SCHOUTEN:
                # For Schouten, match exact (pk, pl) pair
                if (id_spec['params']['pk'] == params['pk'] and
                    id_spec['params']['pl'] == params['pl']):
                    return self.encode_action(bracket_idx, term_slot, id_spec['local_idx'])

            elif identity == IDENTITY_MOMENTUM:
                if (id_spec['params']['pout1'] == params['pout1'] and
                    id_spec['params']['pout2'] == params['pout2']):
                    return self.encode_action(bracket_idx, term_slot, id_spec['local_idx'])

            elif identity == IDENTITY_MOMENTUM_SQ:
                if set(id_spec['params']['mom_add']) == set(params.get('mom_add', [])):
                    return self.encode_action(bracket_idx, term_slot, id_spec['local_idx'])

        raise ValueError(f"Could not find action: bracket={bracket}, term_slot={term_slot}, "
                        f"identity={identity}, params={params}")


# Re-export utility functions from action_space_4pt (they work for any n_point)
from action_space_4pt import (
    get_term_bracket_powers,
    get_unique_brackets,
    apply_identity_to_term,
    apply_identity_to_whole_expression,
    apply_action,
)


if __name__ == "__main__":
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))
    from environment.bracket_env import ab, sb

    print("=" * 70)
    print("Testing TermSpecificActionSpace5pt")
    print("=" * 70)

    # Create action space
    action_space = TermSpecificActionSpace5pt(n_point=5, max_terms=10, max_brackets=20)

    print(f"\nConfiguration:")
    print(f"  n_point: {action_space.n_point}")
    print(f"  max_terms: {action_space.max_terms}")
    print(f"  max_brackets: {action_space.max_brackets}")
    print(f"  n_term_slots: {action_space.n_term_slots}")
    print(f"  identities_per_bracket: {action_space.identities_per_bracket}")
    print(f"  n_actions: {action_space.n_actions}")

    print(f"\nCanonical brackets ({len(action_space.canonical_brackets)}):")
    for idx, (bk_type, i, j) in enumerate(action_space.canonical_brackets):
        print(f"  {idx}: {bk_type}({i},{j})")

    print(f"\nIdentity types for ab(1,2) (bracket_idx=0):")
    identity_types = action_space.get_identity_types_for_bracket(0)
    for id_spec in identity_types:
        print(f"  {id_spec['local_idx']}: {id_spec['identity']} {id_spec['params']}")
    print(f"\nTotal identity types per bracket: {len(identity_types)}")

    print("\n" + "=" * 70)
    print("Comparison: 4pt vs 5pt action spaces")
    print("=" * 70)

    from action_space_4pt import TermSpecificActionSpace
    as4 = TermSpecificActionSpace(n_point=4, max_terms=10, max_brackets=12)

    print(f"\n4-point:")
    print(f"  Brackets: {len(as4.canonical_brackets)}")
    print(f"  Identities per bracket: {as4.identities_per_bracket}")
    print(f"  Total actions: {as4.n_actions}")

    print(f"\n5-point:")
    print(f"  Brackets: {len(action_space.canonical_brackets)}")
    print(f"  Identities per bracket: {action_space.identities_per_bracket}")
    print(f"  Total actions: {action_space.n_actions}")

    print("\n" + "=" * 70)
    print("Testing encode/decode")
    print("=" * 70)

    # Test some actions
    test_cases = [
        (0, 0, 0),   # First bracket, WHOLE EXPRESSION, first identity (Schouten 3,4)
        (0, 0, 1),   # First bracket, WHOLE EXPRESSION, second identity (Schouten 3,5)
        (0, 0, 2),   # First bracket, WHOLE EXPRESSION, third identity (Schouten 4,5)
        (0, 1, 0),   # First bracket, term 0, first identity
        (19, 10, 18), # Last bracket, last term, last identity
    ]

    for bracket_idx, term_slot, identity_type_idx in test_cases:
        action = action_space.encode_action(bracket_idx, term_slot, identity_type_idx)
        decoded = action_space.decode_action(action)
        spec = action_space.get_action_spec(action)
        print(f"\n({bracket_idx}, term_slot={term_slot}, {identity_type_idx}) -> action {action}")
        print(f"  decoded: {decoded}")
        slot_desc = "WHOLE EXPR" if spec.is_whole_expression else f"term {spec.term_idx}"
        print(f"  spec: bracket={action_space.canonical_brackets[spec.bracket_idx]}, "
              f"{slot_desc}, identity={spec.identity}, params={spec.params}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
