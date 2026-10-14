"""Case-sensitive candidate labels; the model must support one-token outputs."""

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
LABEL_CAPACITY = len(LABELS)


def labels_for(count: int) -> tuple[str, ...]:
    if not 2 <= count <= LABEL_CAPACITY:
        raise ValueError(f"Candidate count must be between 2 and {LABEL_CAPACITY}")
    return tuple(LABELS[:count])
