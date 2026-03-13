"""
Extended action space v3: Term-specific identity application for single-instance substitution.

This is a complete redesign of the action space to properly handle cases where we need
to apply an identity to a single instance of a bracket in a specific term.

Key design:
- 12 brackets for 4-point amplitudes: C(4,2) * 2 = 6 pairs * 2 types (ab/sb) = 12
- 11 term slots: 1 "whole expression" + 10 individual terms
- 11 identity types per bracket (1 Schouten + 6 Momentum + 4 Momentum_sq)
- Total: 12 * 11 * 11 = 1452 actions

Action encoding:
    action = bracket_idx * (n_term_slots * identities_per_bracket) + term_slot * identities_per_bracket + identity_type_idx

Where:
- bracket_idx: 0-11 (which of the 12 possible brackets)
- term_slot: 0-10 (0 = entire expression, 1-10 = specific term index 0-9)
- identity_type_idx: 0-10 (which identity variant)

Two action types:
1. term_slot=0 ("whole expression"): Replace ALL occurrences of the bracket in the ENTIRE expression
2. term_slot=1-10 ("single term"): Replace ONE instance (one power) of the bracket in term (term_slot-1)

Identity types for each bracket (i, j) where n_point=4:
  other_momenta = [k for k in 1..4 if k not in (i, j)]  # 2 momenta

  Schouten (1 type): only one distinct action (swapping pk/pl gives identical result)
    - pk, pl = min(other), max(other)  # canonical ordering

  Momentum (6 types): pout1 from bracket args, pout2 from {pout1} + other_momenta
    - For each pout1 in (i, j): 2 choices
    - For each pout2 in {pout1, other[0], other[1]}: 3 choices
    - Total: 2 * 3 = 6
    - Note: pout2 can equal pout1 (gives multi-term expansion)
    - Note: pout2 cannot be the OTHER bracket arg (gives zero denominator)

  Momentum_sq (4 types): all subsets of other_momenta as mom_add
    - [] (empty)
    - [other[0]]
    - [other[1]]
    - [other[0], other[1]]
    - Total: 2^2 = 4

Total per bracket: 1 + 6 + 4 = 11 identity types
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
    bracket_idx: int  # Index in canonical bracket ordering (0-11 for 4-point)
    term_slot: int    # 0 = whole expression, 1-10 = specific term (term_slot-1)
    identity: str     # 'schouten', 'momentum', or 'momentum_sq'
    identity_type_idx: int  # Index within identity types for this bracket (0-10)
    params: Dict[str, Any]  # Identity-specific parameters

    @property
    def is_whole_expression(self) -> bool:
        """True if this action applies to the whole expression."""
        return self.term_slot == 0

    @property
    def term_idx(self) -> Optional[int]:
        """Get the actual term index (0-9) for single-term actions, or None for whole expression."""
        return (self.term_slot - 1) if self.term_slot > 0 else None


class TermSpecificActionSpace:
    """
    Action space with term-specific identity application.

    For 4-point amplitudes:
    - 12 canonical brackets (sorted deterministically)
    - 11 term slots: 1 "whole expression" (slot 0) + 10 individual terms (slots 1-10)
    - 11 identity types per bracket (1 Schouten + 6 Momentum + 4 Momentum_sq)
    - Total: 12 * 11 * 11 = 1452 actions

    Two action types:
    1. term_slot=0: Apply identity to ALL occurrences in ENTIRE expression
    2. term_slot=1-10: Apply identity to ONE instance in term (term_slot-1)
    """

    def __init__(self, n_point: int = 4, max_terms: int = 10, max_brackets: int = 12):
        """
        Args:
            n_point: Number of external momenta (default 4)
            max_terms: Maximum number of terms (default 10)
            max_brackets: Maximum number of brackets (default 12 for 4-point)
        """
        self.n_point = n_point
        self.max_terms = max_terms
        self.max_brackets = max_brackets

        # Number of term slots: 1 (whole expr) + max_terms (individual terms)
        self.n_term_slots = 1 + max_terms

        # For 4-point: 1 + 6 + 4 = 11 identity types per bracket
        self.identities_per_bracket = self._compute_identities_per_bracket()

        # Total action space size
        self.n_actions = max_brackets * self.n_term_slots * self.identities_per_bracket

        # Build canonical bracket ordering
        self.canonical_brackets = self._build_canonical_brackets()

        # Pre-compute identity types for each bracket
        self.identity_types_cache = {}

    def _compute_identities_per_bracket(self) -> int:
        """Compute number of identity types per bracket."""
        # For n_point=4: 1 Schouten + 6 Momentum + 4 Momentum_sq = 11
        n_other = self.n_point - 2  # Other momenta besides bracket args
        n_schouten = 1 if n_other >= 2 else 0  # Only 1 distinct Schouten (swapping pk/pl gives same result)
        n_momentum = 2 * (1 + n_other)  # 2 pout1 choices * (1 + n_other) pout2 choices (pout2 can equal pout1)
        n_momentum_sq = 2 ** n_other  # All subsets of other momenta
        return n_schouten + n_momentum + n_momentum_sq

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
            Canonical index (0-11 for 4-point)
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
            bracket_idx: Canonical index (0-11 for 4-point)

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
            - 'local_idx': index within this bracket's identities (0-10)
        """
        if bracket_idx in self.identity_types_cache:
            return self.identity_types_cache[bracket_idx]

        bk_type, i, j = self.canonical_brackets[bracket_idx]
        bracket_args = (i, j)

        # Other momenta (not in bracket)
        other_momenta = [k for k in range(1, self.n_point + 1) if k not in bracket_args]

        identity_types = []

        # Schouten identity: only 1 distinct action (swapping pk/pl gives identical result)
        # Use canonical ordering: pk < pl
        if len(other_momenta) >= 2:
            pk, pl = min(other_momenta), max(other_momenta)
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
            bracket_idx: Canonical bracket index (0-11)
            term_slot: 0 = whole expression, 1-10 = specific term (term_slot-1)
            identity_type_idx: Identity type index (0-10)

        Returns:
            Action index (0-1451 for default config)
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


def get_term_bracket_powers(expr: sp.Expr, numerator_only: bool = True) -> List[Dict[sp.Function, int]]:
    """
    Get bracket powers for each term in the expression.

    Args:
        expr: SymPy expression
        numerator_only: If True, only look at numerator

    Returns:
        List of dicts, one per term, mapping bracket -> power in that term
    """
    if expr == 0:
        return []

    if numerator_only:
        num, _ = sp.fraction(expr)
    else:
        num = expr

    num_expanded = sp.expand(num)

    if isinstance(num_expanded, sp.Add):
        terms = list(num_expanded.args)
    else:
        terms = [num_expanded]

    result = []
    for term in terms:
        term_powers = {}
        _collect_bracket_powers_from_term(term, term_powers)
        result.append(term_powers)

    return result


def _collect_bracket_powers_from_term(term: sp.Expr, powers: Dict[sp.Function, int]):
    """Helper to collect bracket powers from a single term."""
    if isinstance(term, sp.Mul):
        for factor in term.args:
            _collect_bracket_powers_from_term(factor, powers)
    elif isinstance(term, sp.Pow):
        base, exp = term.as_base_exp()
        if isinstance(base, sp.Function) and base.func.__name__ in ['ab', 'sb']:
            power = int(exp) if isinstance(exp, sp.Integer) else 1
            if power > 0:
                powers[base] = powers.get(base, 0) + power
    elif isinstance(term, sp.Function) and term.func.__name__ in ['ab', 'sb']:
        powers[term] = powers.get(term, 0) + 1


def get_unique_brackets(expr: sp.Expr, numerator_only: bool = True) -> List[sp.Function]:
    """
    Get unique brackets from expression, sorted canonically.

    Args:
        expr: SymPy expression
        numerator_only: If True, only look at numerator

    Returns:
        List of unique brackets, sorted by (type_name, sorted_args)
    """
    if expr == 0:
        return []

    if numerator_only:
        num, _ = sp.fraction(expr)
    else:
        num = expr

    brackets = set()
    for atom in num.atoms(sp.Function):
        if atom.func.__name__ in ['ab', 'sb']:
            brackets.add(atom)

    # Sort canonically
    def sort_key(bk):
        return (bk.func.__name__, tuple(sorted(bk.args)))

    return sorted(brackets, key=sort_key)


def apply_identity_to_term(
    expr: sp.Expr,
    term_idx: int,
    bracket: sp.Function,
    identity_result: sp.Expr,
    power_to_apply: int = 1,
    numerator_only: bool = True
) -> sp.Expr:
    """
    Apply an identity to a specific term in the expression.

    This replaces 'bracket' in the specified term with 'identity_result'.
    If the bracket appears with power > 1 in that term, we replace
    only 'power_to_apply' instances.

    Args:
        expr: SymPy expression
        term_idx: Which term to modify (0-indexed)
        bracket: The bracket to substitute
        identity_result: What to replace bracket with
        power_to_apply: How many instances to substitute (default 1)
        numerator_only: If True, only modify numerator

    Returns:
        Modified expression
    """
    if numerator_only:
        num, denom = sp.fraction(expr)
    else:
        num = expr
        denom = sp.Integer(1)

    num_expanded = sp.expand(num)

    if isinstance(num_expanded, sp.Add):
        terms = list(num_expanded.args)
    else:
        terms = [num_expanded]

    if term_idx < 0 or term_idx >= len(terms):
        # Invalid term index, return unchanged
        return expr

    # Get power of bracket in target term
    term_powers = {}
    _collect_bracket_powers_from_term(terms[term_idx], term_powers)
    power_in_term = term_powers.get(bracket, 0)

    if power_in_term < power_to_apply:
        # Not enough instances to substitute
        return expr

    # Apply substitution to target term
    target_term = terms[term_idx]
    remaining_power = power_in_term - power_to_apply

    if remaining_power == 0:
        # Replace all instances: bracket^n -> identity_result^n
        old_factor = bracket ** power_to_apply if power_to_apply > 1 else bracket
        new_factor = identity_result ** power_to_apply if power_to_apply > 1 else identity_result
    else:
        # Partial replacement: bracket^n -> bracket^(n-k) * identity_result^k
        old_factor = bracket ** power_in_term
        new_factor = (bracket ** remaining_power) * (identity_result ** power_to_apply)

    new_term = target_term.subs(old_factor, new_factor)

    # Rebuild expression
    new_terms = terms[:term_idx] + [new_term] + terms[term_idx + 1:]
    new_num = sum(new_terms) if new_terms else sp.Integer(0)

    return sp.cancel(new_num / denom)


def apply_identity_to_whole_expression(
    expr: sp.Expr,
    bracket: sp.Function,
    identity_result: sp.Expr,
    numerator_only: bool = True
) -> sp.Expr:
    """
    Apply an identity to ALL occurrences of a bracket in the ENTIRE expression.

    This is the "whole expression" action (term_slot=0), which replaces every
    occurrence of the bracket with the identity result.

    Args:
        expr: SymPy expression
        bracket: The bracket to substitute
        identity_result: What to replace bracket with
        numerator_only: If True, only modify numerator

    Returns:
        Modified expression
    """
    if numerator_only:
        num, denom = sp.fraction(expr)
    else:
        num = expr
        denom = sp.Integer(1)

    # Replace all occurrences in numerator
    new_num = num.subs(bracket, identity_result)
    new_num = sp.expand(new_num)

    return sp.cancel(new_num / denom)


def apply_action(
    expr: sp.Expr,
    action_spec: ActionSpec,
    bracket: sp.Function,
    identity_result: sp.Expr,
    numerator_only: bool = True
) -> sp.Expr:
    """
    Apply an action to an expression based on the action specification.

    This is a convenience function that dispatches to the appropriate
    apply function based on term_slot.

    Args:
        expr: SymPy expression
        action_spec: ActionSpec from action_space.get_action_spec()
        bracket: The bracket to substitute
        identity_result: What to replace bracket with
        numerator_only: If True, only modify numerator

    Returns:
        Modified expression
    """
    if action_spec.is_whole_expression:
        # term_slot=0: Apply to entire expression
        return apply_identity_to_whole_expression(
            expr, bracket, identity_result, numerator_only
        )
    else:
        # term_slot=1-10: Apply to specific term (term_slot-1)
        term_idx = action_spec.term_idx
        return apply_identity_to_term(
            expr, term_idx, bracket, identity_result,
            power_to_apply=1, numerator_only=numerator_only
        )


if __name__ == "__main__":
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))
    from environment.bracket_env import ab, sb

    print("=" * 70)
    print("Testing TermSpecificActionSpace")
    print("=" * 70)

    # Create action space
    action_space = TermSpecificActionSpace(n_point=4, max_terms=10, max_brackets=12)

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

    print("\n" + "=" * 70)
    print("Testing encode/decode with term_slot semantics")
    print("=" * 70)

    # Test some actions
    # term_slot=0: whole expression, term_slot=1-10: specific term
    test_cases = [
        (0, 0, 0),   # First bracket, WHOLE EXPRESSION, first identity
        (0, 1, 0),   # First bracket, term 0 (single instance), first identity
        (5, 0, 7),   # Middle bracket, WHOLE EXPRESSION
        (5, 5, 7),   # Middle bracket, term 4 (single instance)
        (11, 10, 10), # Last bracket, term 9 (single instance), last identity
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
    print("Testing find_action_index")
    print("=" * 70)

    # Find action for ab(1,2), WHOLE EXPRESSION, schouten with pk=3, pl=4
    bracket = ab(1, 2)
    action = action_space.find_action_index(
        bracket=bracket,
        term_slot=0,  # Whole expression
        identity=IDENTITY_SCHOUTEN,
        params={'pk': 3, 'pl': 4}
    )
    print(f"\nab(1,2), WHOLE EXPR, schouten(pk=3, pl=4) -> action {action}")
    spec = action_space.get_action_spec(action)
    print(f"  spec: {spec}")

    # Find action for ab(1,2), term 2 (term_slot=3), schouten
    action = action_space.find_action_index(
        bracket=bracket,
        term_slot=3,  # Specific term (term_idx=2)
        identity=IDENTITY_SCHOUTEN,
        params={'pk': 3, 'pl': 4}
    )
    print(f"\nab(1,2), term 2, schouten(pk=3, pl=4) -> action {action}")
    spec = action_space.get_action_spec(action)
    print(f"  spec: {spec}")

    # Find action for sb(2,4), term 5, momentum_sq with mom_add=[1,3]
    bracket = sb(2, 4)
    action = action_space.find_action_index(
        bracket=bracket,
        term_slot=6,  # Specific term (term_idx=5)
        identity=IDENTITY_MOMENTUM_SQ,
        params={'mom_add': [1, 3]}
    )
    print(f"\nsb(2,4), term 5, momentum_sq(mom_add=[1,3]) -> action {action}")
    spec = action_space.get_action_spec(action)
    print(f"  spec: {spec}")

    print("\n" + "=" * 70)
    print("Testing action application (whole expr vs single term)")
    print("=" * 70)

    # Create a test expression with multiple terms
    func_dict = {'ab': ab, 'sb': sb}
    expr = sp.parse_expr("ab(1,2)**2*sb(3,4) + ab(1,2)*ab(3,4) + ab(1,2)**3", local_dict=func_dict)
    print(f"\nOriginal expression: {expr}")

    # Get term bracket powers
    term_powers = get_term_bracket_powers(expr)
    print(f"\nTerm bracket powers:")
    for i, tp in enumerate(term_powers):
        print(f"  term {i}: {tp}")

    # Mock identity result: ab(3,4)*sb(1,2)
    identity_result = sp.parse_expr("ab(3,4)*sb(1,2)", local_dict=func_dict)
    print(f"\nIdentity result: ab(1,2) -> {identity_result}")

    # Test 1: Apply to WHOLE EXPRESSION (replaces ALL ab(1,2))
    new_expr_whole = apply_identity_to_whole_expression(expr, bracket=ab(1, 2),
                                                         identity_result=identity_result)
    print(f"\nAfter applying to WHOLE EXPRESSION (all occurrences):")
    print(f"  {new_expr_whole}")

    # Test 2: Apply to single instance in term 0 (which has ab(1,2)^2)
    new_expr_term0 = apply_identity_to_term(expr, term_idx=0, bracket=ab(1, 2),
                                             identity_result=identity_result, power_to_apply=1)
    print(f"\nAfter applying to term 0 only (1 instance of ab(1,2)^2):")
    print(f"  {new_expr_term0}")

    # Test 3: Apply to single instance in term 2 (which has ab(1,2)^3)
    new_expr_term2 = apply_identity_to_term(expr, term_idx=2, bracket=ab(1, 2),
                                             identity_result=identity_result, power_to_apply=1)
    print(f"\nAfter applying to term 2 only (1 instance of ab(1,2)^3):")
    print(f"  {new_expr_term2}")

    # Test 4: Use apply_action convenience function
    print("\n" + "-" * 40)
    print("Testing apply_action convenience function:")

    # Create an ActionSpec for whole expression
    spec_whole = ActionSpec(
        bracket_idx=0, term_slot=0, identity=IDENTITY_SCHOUTEN,
        identity_type_idx=0, params={'pk': 3, 'pl': 4}
    )
    print(f"\n  spec_whole.is_whole_expression = {spec_whole.is_whole_expression}")
    print(f"  spec_whole.term_idx = {spec_whole.term_idx}")

    # Create an ActionSpec for specific term
    spec_term = ActionSpec(
        bracket_idx=0, term_slot=3, identity=IDENTITY_SCHOUTEN,
        identity_type_idx=0, params={'pk': 3, 'pl': 4}
    )
    print(f"\n  spec_term.is_whole_expression = {spec_term.is_whole_expression}")
    print(f"  spec_term.term_idx = {spec_term.term_idx}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
