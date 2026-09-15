"""The single visual language shared by every figure and render in this repo.

Colours are BGR (cv2 convention, matching `tracking.viz.video.write_video`'s
input contract). Meanings are fixed across the whole repo so figures read
together: white/cyan = observed/detector, green = fit, orange = fly1, grey =
mask/mesh. `fly_colour` gives fly0 cyan / fly1 orange so two animals are told
apart at a glance under `colour_by="fly"`; the anatomy groups (head /
wings+scutellum / abdomen / legs) are for `colour_by="group"`, which is what
you want on a single-fly recording where the question is anatomy rather than
identity.

`keypoint_groups` and `leg_chains` are derived from a keypoint `Order` (or any
name sequence), not from a hardcoded index list, so they work for any anatomy
config's keypoint ordering -- see CLAUDE.md's keypoint-order warning.
"""

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
    """Partition every keypoint NAME into exactly one anatomy group.

    Groups: "head" (antennae/eyes), "wings" (wings + scutellum), "abdomen",
    "legs" (T1-3, L/R). Returns name lists (never indices) so callers never
    have to re-derive a mapping back to a keypoint order.
    """
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
    """Proximal->distal NAME chains, one or more per leg, six legs total.

    A leg missing only its PROXIMAL segment(s) (e.g. no `T2L_ThxCx` in this
    repo's 50-keypoint set) simply starts its chain later -- the remaining
    segments stay in anatomical order and nothing is fabricated. But a leg
    missing an INTERIOR segment (e.g. `FeTi` present-before and present-after
    are not adjacent in the real anatomy once `FeTi` itself is absent) BREAKS
    the chain into separate sub-chains instead of splicing across the gap: a
    drawn segment that bridges a joint the anatomy config does not have is a
    fabricated segment, and CLAUDE.md rates that as worse than a gap. This
    function is only ever asked to draw ADJACENT pairs from within one
    returned chain, so the caller never has to know which kind of gap it was.
    """
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
                # An interior gap: end the current sub-chain here rather than
                # splicing the next present segment onto it. `current` can
                # already be empty here (two consecutive interior gaps) --
                # never append an empty sub-chain.
                if current:
                    chains.append(current)
                current = []
        if current:
            chains.append(current)
    return chains
