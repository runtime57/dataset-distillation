import torch


def _prefix_hash_group_ids(input_ids, num_groups, prefix_length, token_offset):
    token_offset = int(token_offset)
    prefix_length = int(prefix_length)
    if token_offset < 0:
        raise ValueError(f"token_offset must be non-negative, got {token_offset}.")
    if prefix_length <= 0:
        raise ValueError(f"prefix_length must be positive, got {prefix_length}.")
    if token_offset >= input_ids.shape[1]:
        raise ValueError(
            f"token_offset={token_offset} is outside sequence length "
            f"{input_ids.shape[1]}."
        )

    end = min(input_ids.shape[1], token_offset + prefix_length)
    tokens = input_ids[:, token_offset:end].long().cpu()
    weights = torch.arange(1, tokens.shape[1] + 1, dtype=torch.long)
    weights = weights * 1_000_003
    return ((tokens + 1) * weights).sum(dim=1).remainder(num_groups)


def _group_ids_from_tokens(input_ids, method, num_groups, prefix_length, token_offset):
    method = str(method)
    if method == "prefix_hash":
        return _prefix_hash_group_ids(
            input_ids=input_ids,
            num_groups=num_groups,
            prefix_length=prefix_length,
            token_offset=token_offset,
        )
    if method == "first_token":
        token_offset = int(token_offset)
        if token_offset >= input_ids.shape[1]:
            raise ValueError(
                f"token_offset={token_offset} is outside sequence length "
                f"{input_ids.shape[1]}."
            )
        return input_ids[:, token_offset].long().cpu().remainder(num_groups)
    raise ValueError(
        f"Unknown token grouping method {method!r}. "
        "Expected 'prefix_hash' or 'first_token'."
    )


def _balanced_group_ids(num_items, num_groups):
    return torch.arange(int(num_items), dtype=torch.long).remainder(int(num_groups))


def _indices_by_group(group_ids, num_groups):
    return [
        torch.where(group_ids == group_id)[0].long().cpu()
        for group_id in range(int(num_groups))
    ]


def _sample_indices(indices, count):
    if indices.numel() == 0:
        raise ValueError("Cannot sample from an empty group.")
    picks = torch.randint(indices.numel(), (int(count),), device=indices.device)
    return indices[picks]


class TextGroupMatcher:
    """
    Fixed pseudo-labels for conditional text matching.

    Real sequences are grouped by a deterministic token signature. Synthetic
    sequences get fixed group labels, so random-init synthetic rows can learn
    different strata instead of all matching one global average.
    """

    def __init__(
        self,
        real_dataset,
        synthetic_data,
        config,
        synthetic_init_tokens=None,
        embedding_weight=None,
    ):
        if not hasattr(real_dataset, "input_ids"):
            raise ValueError(
                "conditional_matching requires the real dataset to expose "
                "an input_ids tensor."
            )

        self.num_groups = int(config.get("num_groups", 8))
        if self.num_groups <= 0:
            raise ValueError(
                f"conditional_matching.num_groups must be positive, got "
                f"{self.num_groups}."
            )
        self.prefix_length = int(config.get("prefix_length", 8))
        self.token_offset = int(config.get("token_offset", 0))
        self.real_batch_size = int(config.get("real_batch_size", 4))
        self.synth_batch_size = int(config.get("synth_batch_size", 4))
        self.groups_per_step = int(config.get("groups_per_step", self.num_groups))

        real_method = str(config.get("real_group_method", config.get("group_method", "prefix_hash")))
        self.real_input_ids = real_dataset.input_ids.long().cpu()
        self.real_group_ids = _group_ids_from_tokens(
            self.real_input_ids,
            method=real_method,
            num_groups=self.num_groups,
            prefix_length=self.prefix_length,
            token_offset=self.token_offset,
        )
        self.real_indices_by_group = _indices_by_group(
            self.real_group_ids,
            self.num_groups,
        )

        synthetic_method = str(config.get("synthetic_group_method", "balanced"))
        self.synthetic_group_ids = self._build_synthetic_group_ids(
            synthetic_method,
            synthetic_data,
            synthetic_init_tokens,
            embedding_weight,
        )
        self.synthetic_indices_by_group = _indices_by_group(
            self.synthetic_group_ids,
            self.num_groups,
        )

        self.available_groups = torch.tensor(
            [
                group_id
                for group_id in range(self.num_groups)
                if self.real_indices_by_group[group_id].numel() > 0
                and self.synthetic_indices_by_group[group_id].numel() > 0
            ],
            dtype=torch.long,
        )
        if self.available_groups.numel() == 0:
            raise ValueError("conditional_matching produced no non-empty groups.")

    def _build_synthetic_group_ids(
        self,
        method,
        synthetic_data,
        synthetic_init_tokens,
        embedding_weight,
    ):
        if method in ("balanced", "sequence_index_mod"):
            return _balanced_group_ids(synthetic_data.num_sequences, self.num_groups)

        if method in ("init_prefix_hash", "init_first_token"):
            if synthetic_init_tokens is None:
                raise ValueError(
                    f"synthetic_group_method={method!r} requires init tokens."
                )
            token_method = method.removeprefix("init_")
            return _group_ids_from_tokens(
                synthetic_init_tokens.detach().cpu(),
                method=token_method,
                num_groups=self.num_groups,
                prefix_length=self.prefix_length,
                token_offset=self.token_offset,
            )

        if method in ("current_prefix_hash", "current_first_token"):
            token_method = method.removeprefix("current_")
            hard_tokens = synthetic_data.hard_tokens(embedding_weight).detach().cpu()
            return _group_ids_from_tokens(
                hard_tokens,
                method=token_method,
                num_groups=self.num_groups,
                prefix_length=self.prefix_length,
                token_offset=self.token_offset,
            )

        raise ValueError(
            f"Unknown synthetic_group_method {method!r}. Expected one of: "
            "balanced, sequence_index_mod, init_prefix_hash, init_first_token, "
            "current_prefix_hash, current_first_token."
        )

    def sample_groups(self, count):
        count = int(count)
        if count <= 0:
            raise ValueError(f"groups_per_step must be positive, got {count}.")
        if count <= self.available_groups.numel():
            order = torch.randperm(self.available_groups.numel())[:count]
        else:
            order = torch.randint(self.available_groups.numel(), (count,))
        return self.available_groups[order].tolist()

    def sample_real_batch(self, group_id, batch_size, device):
        indices = _sample_indices(
            self.real_indices_by_group[int(group_id)],
            batch_size,
        )
        input_ids = self.real_input_ids[indices].to(device)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids, device=device),
        }

    def sample_synthetic_indices(self, group_id, batch_size, device):
        indices = _sample_indices(
            self.synthetic_indices_by_group[int(group_id)],
            batch_size,
        )
        return indices.to(device)

    def describe(self):
        real_counts = [int(indices.numel()) for indices in self.real_indices_by_group]
        synth_counts = [
            int(indices.numel()) for indices in self.synthetic_indices_by_group
        ]
        return {
            "num_groups": self.num_groups,
            "groups_per_step": self.groups_per_step,
            "real_batch_size": self.real_batch_size,
            "synth_batch_size": self.synth_batch_size,
            "available_groups": int(self.available_groups.numel()),
            "real_group_min": min(real_counts),
            "real_group_max": max(real_counts),
            "synthetic_group_min": min(synth_counts),
            "synthetic_group_max": max(synth_counts),
        }
