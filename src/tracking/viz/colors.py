"""The single visual language shared by every figure and render in this repo."""

from __future__ import annotations

from tracking.io.names import Order, as_order

PALETTE: dict[str, tuple[int, int, int]] = {
    # Shared semantics (BGR).
    "white": (255, 255, 255),  # observed / detector (also: comparison arm)
    "cyan": (255, 255, 0),  # fly0 / observed-detector accent
    "orange": (0, 165, 255),  # fly1
    "green": (0, 255, 0),  # fit
    "grey": (200, 200, 200),  # mask / mesh
    # Anatomy groups, for colour_by="group".
    "head": (0, 0, 255),  # red
    "wings": (0, 255, 255),  # yellow (wings + scutellum)
    "abdomen": (255, 0, 200),  # magenta
    "legs": (255, 140, 0),  # distinct from fly1's orange (0,165,255)
}

# Extra flies beyond fly0/fly1 (not part of the shared 2-fly language, but a
# render must not crash on a 3rd animal).
_EXTRA_FLY_COLOURS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (255, 0, 255),
    (0, 215, 255),
)


def fly_colour(fly: int) -> tuple[int, int, int]:
    """The shared per-fly colour: fly0 cyan, fly1 orange, anything else cycled."""
    if fly == 0:
        return PALETTE["cyan"]
    if fly == 1:
        return PALETTE["orange"]
    return _EXTRA_FLY_COLOURS[(fly - 2) % len(_EXTRA_FLY_COLOURS)]


def keypoint_groups(kp_order: Order | list[str]) -> dict[str, list[str]]:
    """Partition every keypoint NAME into exactly one anatomy group."""
    order = as_order(kp_order)
    groups: dict[str, list[str]] = {"head": [], "wings": [], "abdomen": [], "legs": []}
    for name in order.names:
        leg_prefix = name.split("_", 1)[0]
        if len(leg_prefix) == 3 and leg_prefix[0] == "T" and leg_prefix[1] in "123":
            groups["legs"].append(name)
        elif name.startswith(("Antenna", "Eye")):
            groups["head"].append(name)
        elif name.startswith("Abd"):
            groups["abdomen"].append(name)
        elif name.startswith(("Scutellum", "Wing")):
            groups["wings"].append(name)
        else:
            raise ValueError(
                f"keypoint {name!r} matches no known anatomy group "
                "(head/wings/abdomen/legs); keypoint_groups must partition every name"
            )
    return groups


# Proximal -> distal segment order for one leg.
_LEG_SEGMENTS: tuple[str, ...] = ("ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip")
_LEG_PREFIXES: tuple[str, ...] = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")


def leg_chains(kp_order: Order | list[str]) -> list[list[str]]:
    """Proximal->distal NAME chains, one or more per leg, six legs total."""
    order = as_order(kp_order)
    chains: list[list[str]] = []
    for prefix in _LEG_PREFIXES:
        current: list[str] = []
        started = False  # True once the first present segment of this leg is seen
        for seg in _LEG_SEGMENTS:
            name = f"{prefix}_{seg}"
            if name in order:
                current.append(name)
                started = True
            elif started:
                if current:
                    chains.append(current)
                current = []
        if current:
            chains.append(current)
    return chains
