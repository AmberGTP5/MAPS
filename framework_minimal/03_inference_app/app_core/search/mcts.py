from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from app.search.state import SearchAction, SearchPrior, SearchState, history_to_plan
from app.search.verifier import PlanSearchVerifier


@dataclass
class SearchOutcome:
    plan: list[dict]
    score: float
    iterations: int
    found_terminal: bool
    summary: dict


@dataclass
class MCTSNode:
    state: SearchState
    parent: Optional["MCTSNode"] = None
    action: Optional[SearchAction] = None
    children: list["MCTSNode"] = field(default_factory=list)
    untried_actions: Optional[list[SearchAction]] = None
    visits: int = 0
    value: float = 0.0

    def ucb_score(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)


class PlanSearchEngine:
    def __init__(
        self,
        verifier: Optional[PlanSearchVerifier] = None,
        *,
        max_iterations: int = 96,
        rollout_depth: int = 18,
        exploration_weight: float = 1.25,
        seed: int = 42,
    ) -> None:
        self.verifier = verifier or PlanSearchVerifier()
        self.max_iterations = max_iterations
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.random = random.Random(seed)

    def search(self, prior_plan: Optional[list[dict]] = None) -> SearchOutcome:
        prior = SearchPrior.from_plan(prior_plan or [])
        root = MCTSNode(state=self.verifier.initial_state())

        best_state = root.state
        best_score = self.verifier.score_state(root.state)
        found_terminal = self.verifier.is_terminal(root.state)

        for _ in range(self.max_iterations):
            node = self._select(root, prior)
            if not self.verifier.is_terminal(node.state):
                node = self._expand(node, prior)

            rollout_state = self._rollout(node.state, prior)
            rollout_score = self.verifier.score_state(rollout_state)

            if rollout_score > best_score:
                best_score = rollout_score
                best_state = rollout_state
                found_terminal = self.verifier.is_terminal(rollout_state)

            self._backpropagate(node, rollout_score)

        summary = {
            "iterations": self.max_iterations,
            "found_terminal": found_terminal,
            "best_score": round(best_score, 3),
            "searched_plan_length": len(best_state.history),
            "prior_plan_length": len(prior.sequence),
            "searched_plan": history_to_plan(best_state.history),
        }

        return SearchOutcome(
            plan=history_to_plan(best_state.history),
            score=best_score,
            iterations=self.max_iterations,
            found_terminal=found_terminal,
            summary=summary,
        )

    def _select(self, root: MCTSNode, prior: SearchPrior) -> MCTSNode:
        node = root
        while not self.verifier.is_terminal(node.state):
            if node.untried_actions is None:
                node.untried_actions = self.verifier.candidate_actions(node.state)

            if node.untried_actions:
                return node

            if not node.children:
                return node

            node = max(node.children, key=lambda child: child.ucb_score(self.exploration_weight))

        return node

    def _expand(self, node: MCTSNode, prior: SearchPrior) -> MCTSNode:
        if node.untried_actions is None:
            node.untried_actions = self.verifier.candidate_actions(node.state)

        if not node.untried_actions:
            return node

        depth = len(node.state.history)
        weighted_actions = sorted(
            node.untried_actions,
            key=lambda action: prior.score(action, depth) + self.random.random() * 0.05,
            reverse=True,
        )
        action = weighted_actions[0]
        node.untried_actions.remove(action)

        transition = self.verifier.apply_action(node.state, action)
        if not transition.is_valid:
            return node

        child = MCTSNode(state=transition.next_state, parent=node, action=action)
        node.children.append(child)
        return child

    def _rollout(self, start_state: SearchState, prior: SearchPrior) -> SearchState:
        state = start_state

        for depth_offset in range(self.rollout_depth):
            if self.verifier.is_terminal(state):
                return state

            actions = self.verifier.candidate_actions(state)
            if not actions:
                return state

            best_action = None
            best_next_state = None
            best_value = float("-inf")
            depth = len(state.history) + depth_offset

            for action in actions:
                transition = self.verifier.apply_action(state, action)
                if not transition.is_valid:
                    continue

                value = self.verifier.score_state(transition.next_state) + prior.score(action, depth)
                if value > best_value:
                    best_value = value
                    best_action = action
                    best_next_state = transition.next_state

            if best_action is None or best_next_state is None:
                return state

            state = best_next_state

        return state

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.value += score
            current = current.parent
