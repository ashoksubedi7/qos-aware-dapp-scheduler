MODEL_DEFINITIONS = {
    "M1": {
        "input_dim": 6,
        "hidden_units": (64, 64),
        "output_dim": 22,
        "description": "Context-aware scheduler using backlog and slice SINR.",
    },
    "M2": {
        "input_dim": 7,
        "hidden_units": (64, 64),
        "output_dim": 22,
        "description": (
            "QoS-aware scheduler using backlog, slice SINR, "
            "and URLLC deadline urgency."
        ),
    },
    "M3": {
        "input_dim": 7,
        "hidden_units": (64, 64),
        "output_dim": 22,
        "description": (
            "Counterexample-guided repaired version of M2."
        ),
    },
}


def get_model_definition(name):
    if name not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model: {name}")

    return MODEL_DEFINITIONS[name].copy()
