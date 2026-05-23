"""Social Value Orientation (SVO) helpers.

SVO is a continuous trait theta in [-pi/2, pi/2] that controls how an agent
weights its own welfare versus the team's:

    U_i(s, a; theta) = cos(theta) * r_self(s, a)  +  sin(theta) * r_team(s, a)

Reference convention used throughout the project:
    theta = 0       : selfish      (only self-cost matters)
    theta = pi/4    : prosocial    (equal weight; original BD behavior)
    theta = pi/2    : altruistic   (only team progress matters)
    theta < 0       : competitive  (actively prefers self-gain over team gain)
"""
import math


# Convenient named presets (radians).
SVO_PRESETS = {
    "selfish":     0.0,
    "individualistic": math.pi / 8,    # 22.5 deg
    "prosocial":   math.pi / 4,        # 45   deg (original BD)
    "cooperative": 3 * math.pi / 8,    # 67.5 deg
    "altruistic":  math.pi / 2,        # 90   deg
    "competitive": -math.pi / 4,       # -45  deg
}

# Default if user does not pass --svoX on the CLI: behave like original BD.
DEFAULT_SVO_DEG = 45.0


def parse_svo(value):
    """Parse a CLI SVO value.

    Accepts a float (interpreted as degrees) or a named preset key.
    Returns radians.
    """
    if value is None:
        return math.radians(DEFAULT_SVO_DEG)
    if isinstance(value, (int, float)):
        return math.radians(float(value))
    if isinstance(value, str):
        key = value.strip().lower()
        if key in SVO_PRESETS:
            return SVO_PRESETS[key]
        return math.radians(float(key))
    raise ValueError("Cannot parse SVO value: {!r}".format(value))


def svo_settings(arglist, agent_name):
    """Return this agent's SVO (radians) from the CLI args.

    Mirrors :func:`utils.utils.agent_settings`. Agent names follow the
    ``agent-<int>`` convention from the original repo.
    """
    suffix = agent_name[-1]
    raw = {
        "1": getattr(arglist, "svo1", None),
        "2": getattr(arglist, "svo2", None),
        "3": getattr(arglist, "svo3", None),
        "4": getattr(arglist, "svo4", None),
    }.get(suffix)
    return parse_svo(raw)
