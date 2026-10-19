from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, List

from app.robot_registry import LOCATION_DATABASE
from app.search.state import SearchAction, SearchState


SOURCES = ("Starbucks_Outdoor", "Vending_Machine_2F")
HANDOVER_ZONES = ("Info_Hub_4F", "Elevator_Lobby_4F")
DOG_LOCATIONS = ("Dog_Station_4F", "Info_Hub_4F", "Elevator_Lobby_4F", "User_Desk_Internal")
ELEVATOR_BY_FLOOR = {
    "1F": "Elevator_Lobby_1F",
    "2F": "Elevator_Lobby_2F",
    "4F": "Elevator_Lobby_4F",
    "5F": "Elevator_Lobby_5F",
}


@dataclass(frozen=True)
class SearchConstraints:
    allowed_sources: tuple[str, ...] = SOURCES
    allowed_handover_zones: tuple[str, ...] = HANDOVER_ZONES


@dataclass(frozen=True)
class TransitionResult:
    is_valid: bool
    next_state: SearchState
    reason: str = ""


class PlanSearchVerifier:
    def __init__(self, constraints: SearchConstraints | None = None) -> None:
        self.constraints = constraints or SearchConstraints()

    def set_constraints(self, constraints: SearchConstraints) -> None:
        self.constraints = constraints

    def initial_state(self) -> SearchState:
        return SearchState(
            chassis_location="Robot_Station_5F",
            dog_location="Dog_Station_4F",
            chassis_returned=True,
            dog_returned=True,
        )

    def is_terminal(self, state: SearchState) -> bool:
        return state.delivered and state.chassis_location == "Robot_Station_5F" and state.dog_location == "Dog_Station_4F"

    def score_state(self, state: SearchState) -> float:
        progress = 0.0
        allowed_handover_zones = self.constraints.allowed_handover_zones

        if state.item_holder == "chassis":
            progress += 22.0
        elif state.item_holder == "dog":
            progress += 50.0

        if state.delivered:
            progress += 85.0
        if state.chassis_location == "Robot_Station_5F":
            progress += 8.0
        if state.dog_location == "Dog_Station_4F":
            progress += 8.0
        if state.item_holder == "chassis" and state.chassis_location in allowed_handover_zones:
            progress += 6.0
        if state.item_holder == "chassis" and state.dog_location in allowed_handover_zones:
            progress += 6.0
        if self.is_terminal(state):
            progress += 30.0

        step_penalty = 0.8 * len(state.history)
        return progress - state.path_cost - step_penalty

    def candidate_actions(self, state: SearchState) -> List[SearchAction]:
        actions = set()

        if self.is_terminal(state):
            return []

        if not state.delivered:
            actions.update(self._candidate_pickup_actions(state))
            actions.update(self._candidate_handover_actions(state))
            actions.update(self._candidate_delivery_actions(state))
        else:
            actions.update(self._candidate_return_actions(state))

        if state.item_holder == "dog" and not state.delivered:
            actions.update(self._candidate_return_actions(state, include_dog_only=False))

        return sorted(actions, key=lambda action: (action.robot, action.action, action.target))

    def apply_action(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if action.robot == "Chassis" and action.action == "navigate":
            return self._apply_chassis_navigation(state, action)
        if action.robot == "Chassis" and action.action == "use_elevator":
            return self._apply_elevator(state, action)
        if action.robot == "Dog" and action.action == "navigate":
            return self._apply_dog_navigation(state, action)
        if action.robot == "Arm" and action.action == "grasp":
            return self._apply_grasp(state, action)
        if action.robot == "Arm" and action.action == "transfer_to_dog":
            return self._apply_transfer(state, action)
        if action.robot == "Dog" and action.action == "receive":
            return self._apply_receive(state, action)
        if action.robot == "Dog" and action.action == "deliver":
            return self._apply_deliver(state, action)

        return TransitionResult(False, state, f"Unsupported action: {action}")

    def plan_is_valid(self, plan_steps: Iterable[dict]) -> tuple[bool, str, SearchState]:
        state = self.initial_state()
        for raw_step in plan_steps:
            if not isinstance(raw_step, dict):
                return False, "Plan contains a non-dict step.", state

            action = SearchAction(
                robot=str(raw_step.get("robot", "")),
                action=str(raw_step.get("action", "")),
                target=str(raw_step.get("target", "")),
                details=str(raw_step.get("details", "")),
            )
            result = self.apply_action(state, action)
            if not result.is_valid:
                return False, result.reason, state
            state = result.next_state

        if not self.is_terminal(state):
            return False, "Plan does not satisfy final delivery and return constraints.", state

        return True, "", state

    def _candidate_pickup_actions(self, state: SearchState) -> set[SearchAction]:
        actions = set()
        if state.item_holder != "unpicked":
            return actions

        for source in self.constraints.allowed_sources:
            source_floor = self._floor(source)
            current_floor = self._floor(state.chassis_location)
            current_lobby = ELEVATOR_BY_FLOOR[current_floor]

            if current_floor != source_floor:
                if state.chassis_location != current_lobby:
                    actions.add(
                        SearchAction(
                            robot="Chassis",
                            action="navigate",
                            target=current_lobby,
                            details=f"Move to the {current_floor} elevator lobby before changing floors.",
                        )
                    )
                else:
                    actions.add(
                        SearchAction(
                            robot="Chassis",
                            action="use_elevator",
                            target=source_floor,
                            details=f"Use the elevator to reach {source_floor} for coffee pickup.",
                        )
                    )
                continue

            if state.chassis_location != source:
                actions.add(
                    SearchAction(
                        robot="Chassis",
                        action="navigate",
                        target=source,
                        details=f"Navigate to {source} as a candidate coffee pickup source.",
                    )
                )
            else:
                actions.add(
                    SearchAction(
                        robot="Arm",
                        action="grasp",
                        target="coffee",
                        details=f"Grasp the coffee at {source}.",
                    )
                )

        return actions

    def _candidate_handover_actions(self, state: SearchState) -> set[SearchAction]:
        actions = set()

        if state.item_holder != "chassis":
            return actions

        if state.chassis_location in self.constraints.allowed_handover_zones and state.dog_location == state.chassis_location:
            actions.add(
                SearchAction(
                    robot="Arm",
                    action="transfer_to_dog",
                    target="coffee",
                    details=f"Transfer the coffee to Dog at {state.chassis_location}.",
                )
            )

        for zone in self.constraints.allowed_handover_zones:
            if self._floor(state.chassis_location) != "4F":
                current_floor = self._floor(state.chassis_location)
                current_lobby = ELEVATOR_BY_FLOOR[current_floor]
                if state.chassis_location != current_lobby:
                    actions.add(
                        SearchAction(
                            robot="Chassis",
                            action="navigate",
                            target=current_lobby,
                            details=f"Move to the {current_floor} elevator lobby before heading to 4F.",
                        )
                    )
                else:
                    actions.add(
                        SearchAction(
                            robot="Chassis",
                            action="use_elevator",
                            target="4F",
                            details="Use the elevator to bring the coffee to 4F for handover.",
                        )
                    )
            elif state.chassis_location != zone:
                actions.add(
                    SearchAction(
                        robot="Chassis",
                        action="navigate",
                        target=zone,
                        details=f"Bring the coffee to handover zone {zone}.",
                    )
                )

            if state.dog_location != zone:
                actions.add(
                    SearchAction(
                        robot="Dog",
                        action="navigate",
                        target=zone,
                        details=f"Move Dog to handover zone {zone}.",
                    )
                )

        return actions

    def _candidate_delivery_actions(self, state: SearchState) -> set[SearchAction]:
        actions = set()
        if state.item_holder == "dog" and not state.delivered:
            if state.dog_location != "User_Desk_Internal":
                actions.add(
                    SearchAction(
                        robot="Dog",
                        action="navigate",
                        target="User_Desk_Internal",
                        details="Dog enters the narrow corridor to approach the final delivery location.",
                    )
                )
            else:
                actions.add(
                    SearchAction(
                        robot="Dog",
                        action="deliver",
                        target="coffee",
                        details="Dog delivers the coffee at User_Desk_Internal.",
                    )
                )
        return actions

    def _candidate_return_actions(self, state: SearchState, include_dog_only: bool = True) -> set[SearchAction]:
        actions = set()

        if include_dog_only and state.dog_location != "Dog_Station_4F":
            actions.add(
                SearchAction(
                    robot="Dog",
                    action="navigate",
                    target="Dog_Station_4F",
                    details="Return Dog to its base station after delivery.",
                )
            )

        if state.chassis_location != "Robot_Station_5F":
            current_floor = self._floor(state.chassis_location)
            current_lobby = ELEVATOR_BY_FLOOR[current_floor]
            if current_floor != "5F":
                if state.chassis_location != current_lobby:
                    actions.add(
                        SearchAction(
                            robot="Chassis",
                            action="navigate",
                            target=current_lobby,
                            details=f"Return to the {current_floor} elevator lobby before going back to 5F.",
                        )
                    )
                else:
                    actions.add(
                        SearchAction(
                            robot="Chassis",
                            action="use_elevator",
                            target="5F",
                            details="Use the elevator to return to 5F.",
                        )
                    )
            else:
                actions.add(
                    SearchAction(
                        robot="Chassis",
                        action="navigate",
                        target="Robot_Station_5F",
                        details="Return Chassis to the 5F base station.",
                    )
                )

        return actions

    def _apply_chassis_navigation(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if action.target == "User_Desk_Internal":
            return TransitionResult(False, state, "Chassis cannot enter User_Desk_Internal.")

        if action.target not in LOCATION_DATABASE:
            return TransitionResult(False, state, f"Unknown chassis target: {action.target}")

        if self._floor(action.target) != self._floor(state.chassis_location):
            return TransitionResult(False, state, "Chassis navigation must stay on the current floor.")

        if action.target.startswith("Dog_Station"):
            return TransitionResult(False, state, "Chassis should not navigate into Dog-only station areas.")

        step_cost = self._navigation_cost(state.chassis_location, action.target)
        next_state = state.append(
            action,
            chassis_location=action.target,
            chassis_returned=(action.target == "Robot_Station_5F"),
            path_cost=state.path_cost + step_cost,
        )
        return TransitionResult(True, next_state)

    def _apply_dog_navigation(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if action.target not in DOG_LOCATIONS:
            return TransitionResult(False, state, f"Dog cannot navigate to {action.target}.")

        if self._floor(action.target) != "4F":
            return TransitionResult(False, state, "Dog is confined to the 4F map.")

        step_cost = self._navigation_cost(state.dog_location, action.target)
        updates = {
            "dog_location": action.target,
            "dog_returned": action.target == "Dog_Station_4F",
            "path_cost": state.path_cost + step_cost,
        }

        next_state = state.append(action, **updates)
        return TransitionResult(True, next_state)

    def _apply_elevator(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if action.target not in ELEVATOR_BY_FLOOR:
            return TransitionResult(False, state, f"Unsupported elevator target floor: {action.target}")

        current_floor = self._floor(state.chassis_location)
        current_lobby = ELEVATOR_BY_FLOOR[current_floor]
        if state.chassis_location != current_lobby:
            return TransitionResult(False, state, "Chassis must be at the elevator lobby before changing floors.")

        target_lobby = ELEVATOR_BY_FLOOR[action.target]
        next_state = state.append(
            action,
            chassis_location=target_lobby,
            chassis_returned=target_lobby == "Robot_Station_5F",
            path_cost=state.path_cost + 8.0,
        )
        return TransitionResult(True, next_state)

    def _apply_grasp(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if state.item_holder != "unpicked":
            return TransitionResult(False, state, "Coffee has already been picked up.")
        if state.chassis_location not in SOURCES:
            return TransitionResult(False, state, "Arm can only grasp coffee at a valid pickup source.")
        if state.chassis_location not in self.constraints.allowed_sources:
            return TransitionResult(False, state, "Selected pickup source violates the current task constraints.")

        next_state = state.append(
            action,
            coffee_source=state.chassis_location,
            item_holder="chassis",
            chassis_returned=False,
            path_cost=state.path_cost + 1.0,
        )
        return TransitionResult(True, next_state)

    def _apply_transfer(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if state.item_holder != "chassis":
            return TransitionResult(False, state, "Chassis must hold the coffee before handover.")
        if state.chassis_location != state.dog_location:
            return TransitionResult(False, state, "Dog and Chassis must meet before handover.")
        if state.chassis_location not in HANDOVER_ZONES:
            return TransitionResult(False, state, "Handover must happen at a designated 4F handover zone.")
        if state.chassis_location not in self.constraints.allowed_handover_zones:
            return TransitionResult(False, state, "Selected handover zone violates the current task constraints.")

        next_state = state.append(
            action,
            item_holder="dog",
            path_cost=state.path_cost + 1.0,
        )
        return TransitionResult(True, next_state)

    def _apply_receive(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if state.item_holder != "dog":
            return TransitionResult(False, state, "Dog can only confirm receive after handover is complete.")
        return TransitionResult(True, state.append(action, path_cost=state.path_cost + 0.1))

    def _apply_deliver(self, state: SearchState, action: SearchAction) -> TransitionResult:
        if state.item_holder not in {"dog", "delivered"}:
            return TransitionResult(False, state, "Dog must hold the coffee before delivery.")
        if state.dog_location != "User_Desk_Internal":
            return TransitionResult(False, state, "Dog must be at User_Desk_Internal to deliver the coffee.")

        next_state = state.append(
            action,
            item_holder="delivered",
            delivered=True,
            path_cost=state.path_cost + 0.2,
        )
        return TransitionResult(True, next_state)

    def _floor(self, location_name: str) -> str:
        return str(LOCATION_DATABASE[location_name]["floor"])

    def _navigation_cost(self, start: str, target: str) -> float:
        if start == target:
            return 0.2

        start_data = LOCATION_DATABASE.get(start, {})
        target_data = LOCATION_DATABASE.get(target, {})
        if "coords" in start_data and "coords" in target_data:
            dx = float(start_data["coords"]["x"]) - float(target_data["coords"]["x"])
            dy = float(start_data["coords"]["y"]) - float(target_data["coords"]["y"])
            return round(1.0 + hypot(dx, dy) / 20.0, 3)

        return 2.0
