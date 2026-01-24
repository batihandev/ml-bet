from enum import IntEnum

class Outcome(IntEnum):
    HOME = 0
    DRAW = 1
    AWAY = 2

# Mapping for model training (integer to label)
LABEL_MAPPING = {
    Outcome.HOME: "home",
    Outcome.DRAW: "draw",
    Outcome.AWAY: "away"
}

# Mapping for model training (label to integer)
CLASS_MAPPING = {
    "home": Outcome.HOME,
    "draw": Outcome.DRAW,
    "away": Outcome.AWAY
}

# The target column name for the 1X2 model
TARGET_COL = "ft_1x2_outcome"

# Recommendation thresholds
MIN_PROB = 0.4
MIN_EDGE = 0.05
MIN_EV = 0.1
