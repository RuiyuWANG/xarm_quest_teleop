from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Dict


Factory = Callable[..., Any]
_REGISTRY: Dict[str, Factory] = {}


def register_policy(name: str, factory: Factory) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("policy name must be non-empty")
    _REGISTRY[key] = factory


def _load_builtin(name: str) -> Factory:
    if name == "seeker":
        from xarm_quest_teleop.policy.seeker_policy import SeekerPolicy

        return SeekerPolicy
    if name == "cache_replay":
        from xarm_quest_teleop.policy.cache_action_replay_policy import CacheActionReplayPolicy

        return CacheActionReplayPolicy
    raise KeyError(name)


def _load_entry_points() -> None:
    try:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            selected = eps.select(group="xarm_quest_teleop.policies")
        else:
            selected = eps.get("xarm_quest_teleop.policies", [])
        for ep in selected:
            if ep.name not in _REGISTRY:
                _REGISTRY[ep.name] = ep.load()
    except Exception:
        return


def get_policy_factory(name: str) -> Factory:
    key = str(name).strip()
    if key not in _REGISTRY:
        try:
            _REGISTRY[key] = _load_builtin(key)
        except KeyError:
            _load_entry_points()
    if key not in _REGISTRY:
        known = sorted(set(_REGISTRY) | {"seeker", "cache_replay"})
        raise KeyError(f"Unknown policy {key!r}. Known policies: {known}")
    return _REGISTRY[key]


def create_policy(name: str, **kwargs: Any) -> Any:
    return get_policy_factory(name)(**kwargs)


def available_policies() -> Dict[str, Factory]:
    _load_entry_points()
    for key in ("seeker", "cache_replay"):
        if key not in _REGISTRY:
            try:
                _REGISTRY[key] = _load_builtin(key)
            except Exception:
                pass
    return dict(_REGISTRY)
