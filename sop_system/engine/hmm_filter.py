"""
HMM temporal filter — proper Bayesian time-series smoothing for SOP step detection.

Replaces the ad-hoc confirm buffer with a principled forward-filtering approach:
- States: 0-6 (SOP steps)
- Transition: forward-only, stay-or-advance
- Emission: physical hard constraints × LSTM softmax
- Online filtering: forward algorithm, no need for full Viterbi
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class HMMResult:
    step: int                      # MAP estimate
    confidence: float              # belief mass at MAP step
    beliefs: np.ndarray            # full belief vector (7,)
    is_stable: bool                # belief concentrated on single step


class SOPHMMFilter:
    """Online HMM filter for SOP step detection."""

    def __init__(self, n_states: int = 7, stay_prob: float = 0.85):
        self.n_states = n_states
        self.stay_prob = stay_prob

        # Transition matrix: forward-only, stay-or-advance
        self.A = self._build_transition(stay_prob)

        # Prior belief: start at step 0
        self.belief = np.zeros(n_states, dtype=np.float64)
        self.belief[0] = 1.0

        # Physical constraint: observed_step (from physical engine)
        self.phys_step = 0

        # History for stability check
        self.step_history: List[int] = []
        self.current_step = 0

    def _build_transition(self, stay_prob: float) -> np.ndarray:
        """Build forward-only transition matrix A[s_prev, s_cur]."""
        A = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for s in range(self.n_states):
            if s == self.n_states - 1:  # absorbing state (Done)
                A[s, s] = 1.0
            else:
                A[s, s] = stay_prob
                A[s, s + 1] = 1.0 - stay_prob
            # Normalize
            row_sum = A[s, :].sum()
            if row_sum > 0:
                A[s, :] /= row_sum
        return A

    def reset(self):
        self.belief = np.zeros(self.n_states, dtype=np.float64)
        self.belief[0] = 1.0
        self.phys_step = 0
        self.step_history.clear()
        self.current_step = 0

    def update(self, phys_step: int, lstm_probs: np.ndarray,
               phys_placed_this_frame: bool = False) -> HMMResult:
        """
        Update belief with new observations.

        Args:
            phys_step: physical engine's current step (0-5)
            lstm_probs: LSTM softmax probabilities (7,)
            phys_placed_this_frame: object just placed (strong signal)
        """
        self.phys_step = phys_step

        # ── Emission probability P(obs | state) ──
        emission = self._compute_emission(phys_step, lstm_probs, phys_placed_this_frame)

        # ── Prediction: α_pred = A^T @ belief ──
        belief_pred = self.A.T @ self.belief

        # ── Update: α_new = emission * α_pred / Z ──
        belief_new = emission * belief_pred
        Z = belief_new.sum()
        if Z > 0:
            belief_new /= Z
        else:
            # Numerical underflow: fall back to prior
            belief_new = belief_pred / max(belief_pred.sum(), 1e-10)

        self.belief = belief_new

        # ── MAP estimate ──
        map_step = int(np.argmax(self.belief))
        confidence = float(self.belief[map_step])

        # Stability check: is belief concentrated?
        is_stable = confidence > 0.5

        # Track step
        if map_step != self.current_step:
            self.current_step = map_step
        self.step_history.append(map_step)

        # Keep history bounded
        if len(self.step_history) > 200:
            self.step_history = self.step_history[-100:]

        return HMMResult(
            step=map_step,
            confidence=confidence,
            beliefs=self.belief.copy(),
            is_stable=is_stable,
        )

    def _compute_emission(self, phys_step: int, lstm_probs: np.ndarray,
                          placed_this_frame: bool) -> np.ndarray:
        """
        Compute emission probability for each state.

        Combines:
        - Physical hard constraint: P=0 for states < phys_step
        - LSTM softmax: favors states with high LSTM confidence
        - Physical alignment bonus: states matching phys_step get weight boost
        - Placement event: if object just placed, strongly favor phys_step+1
        """
        emission = np.zeros(self.n_states, dtype=np.float64)
        phys_weight = 0.4  # physical evidence weight
        lstm_weight = 0.6   # GRU/LSTM evidence weight (trained on manual labels)

        for s in range(self.n_states):
            # Physical hard constraint: can't be in a state before phys_step
            if s < phys_step:
                emission[s] = 0.0
                continue

            # Physical alignment: bonus for matching phys_step
            if s == phys_step:
                phys_score = 1.0
            elif s == phys_step + 1:
                phys_score = 0.6  # next step is plausible
            else:
                phys_score = 0.1  # far ahead is unlikely

            # LSTM score from softmax
            lstm_score = float(lstm_probs[s]) if s < len(lstm_probs) else 0.01

            # Weighted combination
            emission[s] = phys_weight * phys_score + lstm_weight * lstm_score

            # Floor: small probability for all "ahead" states
            emission[s] = max(emission[s], 0.005)

        # Placement event: strongly boost the target step
        if placed_this_frame:
            target_step = phys_step  # phys_step already incremented after placement
            if 0 <= target_step < self.n_states:
                emission[target_step] = max(emission[target_step], 0.95)

        # Normalize
        Z = emission.sum()
        if Z > 0:
            emission /= Z

        return emission

    def get_belief(self) -> np.ndarray:
        return self.belief.copy()

    def get_top3(self) -> List[Tuple[int, float]]:
        idxs = np.argsort(-self.belief)[:3]
        return [(int(i), float(self.belief[i])) for i in idxs if self.belief[i] > 0.01]
