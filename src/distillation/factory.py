from src.distillation.parameterizations import (
    AnchorSoftTokenDataset,
    ConceptSoftTokenDataset,
    FullSoftTokenDataset,
    GroupedAnchorSoftTokenDataset,
    GumbelTopKSoftTokenDataset,
    SequenceAnchorSoftTokenDataset,
    SequenceConceptSoftTokenDataset,
    SparseSequenceAnchorSoftTokenDataset,
    TopKSoftTokenDataset,
)


def build_synthetic_data(config):
    synthetic_config = config.distillation.synthetic
    parameterization = synthetic_config.get("parameterization", "full")
    common_kwargs = {
        "num_sequences": synthetic_config.num_sequences,
        "sequence_length": synthetic_config.sequence_length,
        "vocab_size": synthetic_config.vocab_size,
        "temperature": synthetic_config.temperature,
        "init_std": synthetic_config.init_std,
    }

    if parameterization == "full":
        return FullSoftTokenDataset(**common_kwargs)
    if parameterization == "topk":
        return TopKSoftTokenDataset(
            **common_kwargs,
            k=synthetic_config.get("topk", 16),
        )
    if parameterization == "topk_gumbel":
        return GumbelTopKSoftTokenDataset(
            **common_kwargs,
            k=synthetic_config.get("topk", 16),
        )
    if parameterization == "anchors":
        return AnchorSoftTokenDataset(
            **common_kwargs,
            num_anchors=synthetic_config.get("num_anchors", 64),
        )
    if parameterization == "grouped_anchors":
        return GroupedAnchorSoftTokenDataset(
            **common_kwargs,
            num_anchors=synthetic_config.get("num_anchors", 64),
            num_groups=synthetic_config.get("num_groups", 8),
        )
    if parameterization == "sequence_anchors":
        return SequenceAnchorSoftTokenDataset(
            **common_kwargs,
            num_anchors=synthetic_config.get("num_anchors", 16),
        )
    if parameterization == "sparse_sequence_anchors":
        return SparseSequenceAnchorSoftTokenDataset(
            **common_kwargs,
            num_anchors=synthetic_config.get("num_anchors", 16),
            anchor_topk=synthetic_config.get("anchor_topk", 8),
        )
    if parameterization == "concepts":
        return ConceptSoftTokenDataset(
            **common_kwargs,
            num_concepts=synthetic_config.get("num_concepts", 64),
            d_model=config.model.d_model,
            logit_scale=synthetic_config.get("concept_logit_scale", 32.0),
            input_mode=synthetic_config.get("concept_input_mode", "probs"),
        )
    if parameterization == "sequence_concepts":
        return SequenceConceptSoftTokenDataset(
            **common_kwargs,
            num_concepts=synthetic_config.get("num_concepts", 16),
            d_model=config.model.d_model,
            logit_scale=synthetic_config.get("concept_logit_scale", 32.0),
            input_mode=synthetic_config.get("concept_input_mode", "probs"),
        )

    raise ValueError(
        "Unknown synthetic parameterization "
        f"{parameterization!r}. Expected one of: "
        "full, topk, topk_gumbel, anchors, grouped_anchors, "
        "sequence_anchors, sparse_sequence_anchors, concepts, "
        "sequence_concepts."
    )
