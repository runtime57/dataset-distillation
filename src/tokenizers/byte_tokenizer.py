class ByteTokenizer:
    """
    Minimal UTF-8 byte tokenizer.

    It keeps the vocabulary small enough for early full-soft-token
    experiments while staying independent from external tokenizer packages.
    """

    pad_token_id = 0
    eos_token_id = 1
    byte_offset = 2
    vocab_size = 258

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        token_ids = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids) -> str:
        byte_values = []
        for token_id in token_ids:
            if token_id in (self.pad_token_id, self.eos_token_id):
                continue
            if token_id >= self.byte_offset:
                byte_values.append(token_id - self.byte_offset)
        return bytes(byte_values).decode("utf-8", errors="replace")
