from transformers import AutoTokenizer


class GPT2BPETokenizer:
    """
    GPT-2 byte-pair tokenizer wrapper with an explicit EOS-as-PAD policy.
    """

    def __init__(self, tokenizer_name="gpt2", cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos and self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids) -> str:
        filtered = [int(token_id) for token_id in token_ids]
        return self.tokenizer.decode(filtered, skip_special_tokens=True)
