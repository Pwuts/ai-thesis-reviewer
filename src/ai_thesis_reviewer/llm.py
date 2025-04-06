import anthropic


class Usage(anthropic.types.Usage):
    @staticmethod
    def null():
        return Usage(input_tokens=0, output_tokens=0)

    @property
    def cost(self) -> float:
        # Claude 3.7 Sonnet pricing
        return (
            self.input_tokens * 3 / 1e6
            + self.output_tokens * 15 / 1e6
            + (self.cache_creation_input_tokens or 0) * 3.75 / 1e6
            + (self.cache_read_input_tokens or 0) * 0.3 / 1e6
        )

    def __add__(self, other: anthropic.types.Usage) -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                (self.cache_creation_input_tokens or 0)
                + (other.cache_creation_input_tokens or 0)
                if self.cache_creation_input_tokens or other.cache_creation_input_tokens
                else None
            ),
            cache_read_input_tokens=(
                (self.cache_read_input_tokens or 0)
                + (other.cache_read_input_tokens or 0)
                if self.cache_read_input_tokens or other.cache_read_input_tokens
                else None
            ),
        )
