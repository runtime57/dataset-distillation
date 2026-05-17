from tokenizers import Tokenizer


class LocalBPETokenizer:
    """
    Lightweight wrapper around a locally trained tokenizers BPE tokenizer.
    """

    def __init__(
        self,
        tokenizer_path,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.pad_token_id = self.tokenizer.token_to_id(pad_token)
        self.eos_token_id = self.tokenizer.token_to_id(eos_token)
        self.unk_token_id = self.tokenizer.token_to_id(unk_token)
        self.vocab_size = self.tokenizer.get_vocab_size()

        if self.pad_token_id is None or self.eos_token_id is None:
            raise ValueError(
                "LocalBPETokenizer requires tokenizer.json to contain "
                f"{pad_token!r} and {eos_token!r} special tokens."
            )

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids) -> str:
        filtered = [
            int(token_id)
            for token_id in token_ids
            if int(token_id) not in {self.pad_token_id, self.eos_token_id}
        ]
        return self.tokenizer.decode(filtered)
