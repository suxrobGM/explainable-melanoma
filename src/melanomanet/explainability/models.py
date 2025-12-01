from attr import dataclass


@dataclass
class ConceptScore:
    """Container for concept importance scores."""

    concept_name: str
    tcav_score: float  # Directional derivative score
    accuracy: float  # CAV classifier accuracy
    p_value: float  # Statistical significance
    is_significant: bool  # True if p_value < 0.05


@dataclass
class FastCAVResult:
    """Container for full FastCAV analysis results."""

    target_class: str
    concept_scores: dict[str, ConceptScore]
    feature_dim: int
    n_samples_used: int
