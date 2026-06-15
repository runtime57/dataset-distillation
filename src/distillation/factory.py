from src.distillation.parameterizations import (
    AnchorSoftTokenDataset,
    ConceptSoftTokenDataset,
    DecoupledGumbelTopKSoftTokenDataset,
    FixedTargetGumbelTopKSoftTokenDataset,
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
    hard_forward = synthetic_config.get("hard_forward", False)

    if parameterization == "full":
        return FullSoftTokenDataset(
            **common_kwargs,
            hard_forward=hard_forward,
        )
    if parameterization == "topk":
        return TopKSoftTokenDataset(
            **common_kwargs,
            k=synthetic_config.get("topk", 16),
            hard_forward=hard_forward,
        )
    if parameterization == "topk_gumbel":
        return GumbelTopKSoftTokenDataset(
            **common_kwargs,
            k=synthetic_config.get("topk", 16),
            gradient_temperature=synthetic_config.get("gradient_temperature"),
            hard_forward=hard_forward,
        )
    if parameterization == "decoupled_topk_gumbel":
        return DecoupledGumbelTopKSoftTokenDataset(
            **common_kwargs,
            k=synthetic_config.get("topk", 16),
            gradient_temperature=synthetic_config.get("gradient_temperature"),
            target_init_std=synthetic_config.get("target_init_std"),
            target_init_confidence=synthetic_config.get("target_init_confidence"),
            hard_forward=hard_forward,
            target_hard_forward=synthetic_config.get("target_hard_forward", False),
        )
    if parameterization == "fixed_target_topk_gumbel":
        return FixedTargetGumbelTopKSoftTokenDataset(
            **common_kwargs,
            k=synthetic_config.get("topk", 16),
            gradient_temperature=synthetic_config.get("gradient_temperature"),
            target_checkpoint_path=synthetic_config.get("target_checkpoint_path"),
            hard_forward=hard_forward,
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
