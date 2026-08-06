"""Module for abstracting tokeinization logic."""

import inspect
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from chonkie.logger import get_logger

if TYPE_CHECKING:
    import tiktoken
    import tokenizers
    import tokie
    import transformers


logger = get_logger(__name__)

_TIKTOKEN_TO_TOKIE_MAPPING = {
    "cl100k_base": "Xenova/gpt-4",
    "o200k_base": "Xenova/gpt-4o",
    "p50k_base": "Xenova/text-davinci-003",
    "gpt2": "openai-community/gpt2",
}
_HUGGINGFACE_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_HUGGINGFACE_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


class TokenizerProtocol(Protocol):
    """Protocol defining the interface for tokenizers.

    Any object implementing these methods can be used as a tokenizer in Chonkie.
    """

    def encode(self, text: str) -> Sequence[int]:
        """Encode text into token IDs.

        Args:
            text: The text to encode.

        Returns:
            Sequence of token IDs.

        """
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: Sequence of token IDs.

        Returns:
            Decoded text string.

        """
        ...

    def tokenize(self, text: str) -> Sequence[str | int]:
        """Tokenize text into tokens.

        Args:
            text: The text to tokenize.

        Returns:
            Sequence of tokens (strings or IDs).

        """
        ...


class Tokenizer(ABC):
    """Base class for tokenizers implementing the TokenizerProtocol.

    This class provides a foundation for creating custom tokenizers with
    vocabulary management and default implementations of common methods.
    """

    def __init__(self) -> None:
        """Initialize the Tokenizer."""
        self.vocab: list[str] = []
        self.token2id: dict[str, int] = defaultdict(self.defaulttoken2id)
        # Note: Using a lambda here would cause pickling issues:
        # self.token2id: dict[str, int] = defaultdict(lambda: len(self.vocab))
        self.token2id[" "]  # Add space to the vocabulary
        self.vocab.append(" ")  # Add space to the vocabulary

    def defaulttoken2id(self) -> int:
        """Return the default token ID.

        This method is used as the default_factory for defaultdict.
        Using a named method instead of a lambda ensures the object can be pickled.
        """
        return len(self.vocab)

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the Tokenizer."""
        return f"{self.__class__.__name__}(vocab_size={len(self.vocab)})"

    def get_vocab(self) -> Sequence[str]:
        """Return the vocabulary."""
        return self.vocab

    def get_token2id(self) -> dict:
        """Return token-to-id mapping."""
        return self.token2id

    @abstractmethod
    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        raise NotImplementedError("Encoding not implemented for base tokenizer.")

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the given tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        raise NotImplementedError("Decoding not implemented for base tokenizer.")

    @abstractmethod
    def tokenize(self, text: str) -> Sequence[str | int]:
        """Tokenize the given text.

        Args:
            text (str): The text to tokenize.

        Returns:
            Sequence of tokens (strings or token IDs)

        """
        raise NotImplementedError("Tokenization not implemented for base tokenizer.")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Default implementation uses tokenize() method.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(self.tokenize(text))

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode a list of texts into tokens.

        Args:
            texts (Sequence[str]): The texts to encode.

        Returns:
            List of encoded sequences

        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode a list of tokens back into text.

        Args:
            token_sequences (Sequence[Sequence[int]]): The tokens to decode.

        Returns:
            List of decoded texts

        """
        return [self.decode(tokens) for tokens in token_sequences]

    def count_tokens_batch(self, texts: Sequence[str]) -> Sequence[int]:
        """Count the number of tokens in a batch of texts.

        Args:
            texts (Sequence[str]): The texts to count tokens in.

        Returns:
            List of token counts

        """
        return [self.count_tokens(text) for text in texts]


class CharacterTokenizer(Tokenizer):
    """Character-based tokenizer."""

    def __repr__(self) -> str:
        """Return a string representation of the CharacterTokenizer."""
        return f"CharacterTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize text into individual characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of characters

        """
        return list(text)

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        encoded = []
        for token in text:
            id = self.token2id[token]
            if id >= len(self.vocab):
                self.vocab.append(token)
            encoded.append(id)
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the given tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        try:
            return "".join([self.vocab[token] for token in tokens])
        except Exception as e:
            raise ValueError(f"Decoding failed. Tokens: {tokens} not found in vocab.") from e

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(text)


class WordTokenizer(Tokenizer):
    """Word-based tokenizer."""

    def __repr__(self) -> str:
        """Return a string representation of the WordTokenizer."""
        return f"WordTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize the given text into words.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of tokens

        """
        return text.split(" ")

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        encoded = []
        for token in self.tokenize(text):
            id = self.token2id[token]
            if id >= len(self.vocab):
                self.vocab.append(token)
            encoded.append(id)
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token ids back to text."""
        try:
            return " ".join([self.vocab[token] for token in tokens])
        except Exception as e:
            raise ValueError(f"Decoding failed. Tokens: {tokens} not found in vocab.") from e

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(self.tokenize(text))


class ByteTokenizer(Tokenizer):
    """Byte-based tokenizer that operates on UTF-8 encoded bytes."""

    def __repr__(self) -> str:
        """Return a string representation of the ByteTokenizer."""
        return f"ByteTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[int]:
        """Tokenize text into individual bytes.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of byte values

        """
        return list(text.encode("utf-8"))

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into byte tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence of byte values

        """
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode byte tokens back into text.

        Args:
            tokens (Sequence[int]): The byte tokens to decode.

        Returns:
            Decoded text

        """
        try:
            return bytes(tokens).decode("utf-8")
        except Exception as e:
            raise ValueError(
                f"Decoding failed. Tokens: {tokens} cannot be decoded as UTF-8.",
            ) from e

    def count_tokens(self, text: str) -> int:
        """Count the number of byte tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of byte tokens

        """
        return len(text.encode("utf-8"))


class RowTokenizer(Tokenizer):
    """Row-based tokenizer that counts lines/rows in text.

    This tokenizer treats each line (separated by newlines) as a token.
    It is primarily useful for table chunking where you want to chunk
    by number of rows rather than by character or subword tokens.
    """

    def __repr__(self) -> str:
        """Return a string representation of the RowTokenizer."""
        return f"RowTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize text into individual lines/rows.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of lines/rows

        """
        if not text:
            return []
        return text.split("\n")

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        encoded = []
        for token in self.tokenize(text):
            id = self.token2id[token]
            if id >= len(self.vocab):
                self.vocab.append(token)
            encoded.append(id)
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the given tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        try:
            return "\n".join([self.vocab[token] for token in tokens])
        except Exception as e:
            raise ValueError(f"Decoding failed. Tokens: {tokens} not found in vocab.") from e

    def count_tokens(self, text: str) -> int:
        """Count the number of rows/lines in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of rows/lines

        """
        if not text:
            return 0
        return len(text.split("\n"))


_chonkie_tokenizer_classes = {
    "character": CharacterTokenizer,
    "word": WordTokenizer,
    "byte": ByteTokenizer,
    "row": RowTokenizer,
}


class InvalidTokenizerError(ValueError):
    """Error raised when a tokenizer can't be loaded."""

    def __init__(self, message: str, *, backend_errors: dict[str, str]) -> None:  # noqa: D107
        super().__init__(message)
        self.backend_errors = backend_errors


def _is_huggingface_offline() -> bool:
    """Return whether Hugging Face offline mode is enabled."""
    return any(
        os.getenv(variable, "").strip().upper() in _HUGGINGFACE_TRUE_VALUES
        for variable in _HUGGINGFACE_OFFLINE_ENV_VARS
    )


def _resolve_huggingface_tokenizer_file(identifier: str) -> Path:
    """Resolve a tokenizer JSON file without making a network request."""
    local_path = Path(identifier).expanduser()
    if local_path.is_file():
        return local_path

    if local_path.is_dir():
        local_tokenizer_file = local_path / "tokenizer.json"
        if local_tokenizer_file.is_file():
            return local_tokenizer_file

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as e:
        raise ImportError(
            "The optional `huggingface-hub` package is required to resolve a cached "
            "tokenizer by model identifier in offline mode. Install it with "
            "`pip install chonkie[hub]`, or pass a local tokenizer.json path."
        ) from e

    cached_file = try_to_load_from_cache(identifier, "tokenizer.json", revision="main")
    if isinstance(cached_file, (str, os.PathLike)):
        cached_path = Path(cached_file)
        if cached_path.is_file():
            return cached_path

    raise FileNotFoundError(
        f"Tokenizer file for {identifier!r} was not found locally or in the "
        "Hugging Face cache; offline mode does not allow Hub downloads."
    )


def _create_auto_tokenizer_from_string(tokenizer: str) -> "AutoTokenizer":
    if tokenizer_cls := _chonkie_tokenizer_classes.get(tokenizer):
        return ChonkieAutoTokenizer(tokenizer_cls())

    backend_errors = {}

    tokie_name = _TIKTOKEN_TO_TOKIE_MAPPING.get(tokenizer)

    if _is_huggingface_offline():
        tokenizer_names = [tokie_name] if tokie_name is not None else []
        if tokenizer not in tokenizer_names:
            tokenizer_names.append(tokenizer)

        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as e:
            backend_errors["tokenizers"] = (
                "The optional `tokenizers` package is required to load a tokenizer "
                "in offline mode. Install it with `pip install chonkie[tokenizers]`."
            )
            raise InvalidTokenizerError(
                f"Tokenizer {tokenizer!r} could not be loaded in offline mode: {backend_errors}",
                backend_errors=backend_errors,
            ) from e

        for name in tokenizer_names:
            try:
                tokenizer_file = _resolve_huggingface_tokenizer_file(name)
                return TokenizersAutoTokenizer(HFTokenizer.from_file(str(tokenizer_file)))
            except Exception as e:
                backend_errors[f"tokenizers ({name})"] = str(e)

        raise InvalidTokenizerError(
            f"Tokenizer {tokenizer!r} could not be loaded in offline mode: {backend_errors}",
            backend_errors=backend_errors,
        )

    from tokie import Tokenizer as TokieTokenizer

    if tokie_name is not None:
        try:
            return TokieAutoTokenizer(TokieTokenizer.from_pretrained(tokie_name))
        except Exception as e:
            backend_errors["tokie (mapped)"] = str(e)

    try:
        return TokieAutoTokenizer(TokieTokenizer.from_pretrained(tokenizer))
    except Exception as e:
        backend_errors["tokie"] = str(e)

    raise InvalidTokenizerError(
        f"Tokenizer {tokenizer!r} could not be loaded: {backend_errors}",
        backend_errors=backend_errors,
    )


class AutoTokenizer:
    """Auto-loading tokenizer interface for Chonkie.

    This class provides automatic loading of tokenizers from various sources
    (string identifiers, HuggingFace models, tiktoken, etc.) and wraps them
    with a unified interface.

    When instantiated, this class automatically returns the appropriate
    backend-specific adapter instance based on the tokenizer type.

    Args:
        tokenizer: Tokenizer identifier or instance.

    Raises:
        ImportError: If the specified tokenizer is not available.

    """

    def __new__(cls, tokenizer: str | Callable | Any = "character") -> "AutoTokenizer":
        """Create and return the appropriate tokenizer adapter instance."""
        # If we're being called on a subclass (adapter), just create it normally
        if cls is not AutoTokenizer:
            return object.__new__(cls)

        # If tokenizer is already an AutoTokenizer, return it as-is
        if isinstance(tokenizer, AutoTokenizer):
            return tokenizer

        if isinstance(tokenizer, Tokenizer):
            return ChonkieAutoTokenizer(tokenizer)

        # Load tokenizer from string if needed
        if isinstance(tokenizer, str):
            return _create_auto_tokenizer_from_string(tokenizer)

        from tokie import Tokenizer as TokieTokenizer

        if isinstance(tokenizer, TokieTokenizer):
            return TokieAutoTokenizer(tokenizer)

        supported_backends = [
            ("transformers", TransformersAutoTokenizer),
            ("tokenizers", TokenizersAutoTokenizer),
            ("tiktoken", TiktokenAutoTokenizer),
        ]
        for backend_name, adapter_class in supported_backends:
            if backend_name in str(type(tokenizer)):
                return adapter_class(tokenizer)

        if callable(tokenizer) or inspect.isfunction(tokenizer) or inspect.ismethod(tokenizer):
            return CallableAutoTokenizer(tokenizer)

        raise ValueError(f"Unsupported tokenizer backend: {type(tokenizer)}")

    def __init__(self, tokenizer: Any):
        """Initialize the adapter with the underlying tokenizer."""
        # Only initialize if we haven't already (avoid re-init during __new__)
        if not hasattr(self, "tokenizer"):
            self.tokenizer = tokenizer

    def encode(self, text: str) -> Sequence[int]:
        """Encode the text into tokens."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the tokens back into text."""
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        # Try using native count_tokens if available, otherwise use encode length
        if hasattr(self.tokenizer, "count_tokens"):
            return self.tokenizer.count_tokens(text)
        return len(self.encode(text))

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode a list of texts into tokens."""
        return self.tokenizer.encode_batch(texts)

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode a list of tokens back into text."""
        return self.tokenizer.decode_batch(token_sequences)

    def count_tokens_batch(self, texts: Sequence[str]) -> Sequence[int]:
        """Count the number of tokens in a batch of texts."""
        # Try using native count_tokens_batch if available, otherwise use list comprehension
        if hasattr(self.tokenizer, "count_tokens_batch"):
            return self.tokenizer.count_tokens_batch(texts)
        return [self.count_tokens(text) for text in texts]


class ChonkieAutoTokenizer(AutoTokenizer):
    """Adapter for chonkie tokenizers."""

    _backend = "chonkie"
    tokenizer: Tokenizer


class TiktokenAutoTokenizer(AutoTokenizer):
    """Adapter for tiktoken tokenizers."""

    _backend = "tiktoken"

    if TYPE_CHECKING:
        tokenizer: tiktoken.Encoding


class TransformersAutoTokenizer(AutoTokenizer):
    """Adapter for HuggingFace `transformers` tokenizers."""

    _backend = "transformers"

    if TYPE_CHECKING:
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast

    def encode(self, text: str) -> Sequence[int]:
        """Encode the text into tokens without special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode texts without special tokens."""
        encoded = self.tokenizer(texts, add_special_tokens=False)
        return encoded["input_ids"]

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode using batch_decode method."""
        return self.tokenizer.batch_decode(
            [list(seq) for seq in token_sequences], skip_special_tokens=True
        )


class TokenizersAutoTokenizer(AutoTokenizer):
    """Adapter for HuggingFace `tokenizers` tokenizers."""

    _backend = "tokenizers"

    if TYPE_CHECKING:
        tokenizer: tokenizers.Tokenizer

    def encode(self, text: str) -> Sequence[int]:
        """Encode text and extract token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode and extract token IDs from each encoding."""
        return [
            encoding.ids
            for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=False)
        ]


class TokieAutoTokenizer(AutoTokenizer):
    """Adapter for tokie tokenizers."""

    _backend = "tokie"

    if TYPE_CHECKING:
        tokenizer: tokie.Tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text and extract token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs back to text."""
        result = self.tokenizer.decode(list(tokens))
        return result if result is not None else ""

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokie's native method."""
        return self.tokenizer.count_tokens(text)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Batch encode texts and extract token IDs."""
        encodings = self.tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [enc.ids for enc in encodings]

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> list[str]:
        """Batch decode token IDs back to text."""
        results = self.tokenizer.decode_batch([list(seq) for seq in token_sequences])
        return [r if r is not None else "" for r in results]

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """Batch count tokens using tokie's native method."""
        return self.tokenizer.count_tokens_batch(list(texts))


class CallableAutoTokenizer(AutoTokenizer):
    """Adapter for user-provided callable token counters."""

    _backend = "callable"
    tokenizer: Callable[[str], int]

    def encode(self, text: str) -> Sequence[int]:
        """Not implemented for callable tokenizers."""
        raise NotImplementedError("Encoding not implemented for callable tokenizers.")

    def decode(self, tokens: Sequence[int]) -> str:
        """Not implemented for callable tokenizers."""
        raise NotImplementedError("Decoding not implemented for callable tokenizers.")

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Not implemented for callable tokenizers."""
        raise NotImplementedError("Batch encoding not implemented for callable tokenizers.")

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Not implemented for callable tokenizers."""
        raise NotImplementedError("Batch decoding not implemented for callable tokenizers.")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return self.tokenizer(text)
