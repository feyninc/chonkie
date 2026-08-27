"""Unit tests for the tokenizer module."""

from typing import Any, Callable

import pytest
from tokie import Tokenizer as TokieTokenizer

from chonkie.tokenizer import (
    AutoTokenizer,
    ByteTokenizer,
    CharacterTokenizer,
    WordTokenizer,
)

tiktoken = pytest.importorskip("tiktoken", reason="tiktoken not installed")
HFTokenizer = pytest.importorskip("tokenizers", reason="tokenizers not installed").Tokenizer
_transformers = pytest.importorskip("transformers", reason="transformers not installed")
HFAutoTokenizer = _transformers.AutoTokenizer
PreTrainedTokenizerFast = _transformers.PreTrainedTokenizerFast


@pytest.fixture
def sample_text() -> str:
    """Fixture to provide sample text for testing."""
    return """The quick brown fox jumps over the lazy dog.
    This classic pangram contains all the letters of the English alphabet.
    It's often used for testing typefaces and keyboard layouts.
    Text chunking, the process you are working on, 
    involves dividing a larger text into smaller, contiguous pieces or 'chunks'.
    This is fundamental in many Natural Language Processing (NLP) tasks.
    For instance, large documents might be chunked into paragraphs or sections 
    before feeding them into a machine learning model due to memory constraints 
    or to process contextually relevant blocks. 
    Other applications include displaying text incrementally in user interfaces 
    or preparing data for certain types of linguistic analysis. 
    Effective chunking might consider sentence boundaries 
    (using periods, question marks, exclamation points), 
    paragraph breaks (often marked by double newlines), 
    or simply aim for fixed-size chunks based on character or word counts. 
    The ideal strategy depends heavily on the specific downstream application. 
    Testing should cover various scenarios, including text with short sentences, 
    long sentences, multiple paragraphs, and potentially unusual punctuation or spacing."""


@pytest.fixture
def sample_text_list() -> list[str]:
    """Fixture to provide a list of sample text for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "This classic pangram contains all the letters of the English alphabet.",
        "It's often used for testing typefaces and keyboard layouts.",
        "Text chunking, the process you are working on, involves dividing a larger text into smaller, contiguous pieces or 'chunks'.",
        "This is fundamental in many Natural Language Processing (NLP) tasks.",
        "For instance, large documents might be chunked into paragraphs or sections before feeding them into a machine learning model due to memory constraints or to process contextually relevant blocks.",
        "Other applications include displaying text incrementally in user interfaces or preparing data for certain types of linguistic analysis.",
        "Effective chunking might consider sentence boundaries (using periods, question marks, exclamation points), paragraph breaks (often marked by double newlines), or simply aim for fixed-size chunks based on character or word counts.",
        "The ideal strategy depends heavily on the specific downstream application.",
        "Testing should cover various scenarios, including text with short sentences, long sentences, multiple paragraphs, and potentially unusual punctuation or spacing.",
    ]


@pytest.fixture
def character_tokenizer() -> CharacterTokenizer:
    """Character tokenizer fixture."""
    return CharacterTokenizer()


@pytest.fixture
def word_tokenizer() -> WordTokenizer:
    """Word tokenizer fixture."""
    return WordTokenizer()


@pytest.fixture
def byte_tokenizer() -> ByteTokenizer:
    """Byte tokenizer fixture."""
    return ByteTokenizer()


@pytest.fixture
def hf_tokenizer() -> HFTokenizer:
    """Create a HuggingFace tokenizer fixture."""
    return HFTokenizer.from_pretrained("gpt2")


@pytest.fixture
def tiktoken_tokenizer() -> tiktoken.Encoding:
    """Create a Tiktoken tokenizer fixture."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def transformers_tokenizer() -> PreTrainedTokenizerFast:
    """Create a Transformer tokenizer fixture."""
    tokenizer: PreTrainedTokenizerFast = HFAutoTokenizer.from_pretrained("gpt2")
    return tokenizer


@pytest.fixture
def callable_tokenizer() -> Callable[[str], int]:
    """Create a callable tokenizer fixture."""
    return lambda text: len(text.split())


@pytest.mark.parametrize(
    "backend_str",
    [
        "hf_tokenizer",
        "tiktoken_tokenizer",
        "transformers_tokenizer",
        "callable_tokenizer",
    ],
)
def test_backend_selection(request: pytest.FixtureRequest, backend_str: str) -> None:
    """Test that the tokenizer correctly selects the backend based on given string."""
    try:
        tokenizer = AutoTokenizer(request.getfixturevalue(backend_str))
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {e}")

    assert tokenizer._backend in [
        "transformers",
        "tokenizers",
        "tiktoken",
        "callable",
    ]


@pytest.mark.parametrize("model_name", ["gpt2", "p50k_base"])
def test_string_init(model_name: str) -> None:
    """Test initialization of tokenizer with different model strings."""
    try:
        tokenizer = AutoTokenizer(model_name)
        assert tokenizer is not None
        assert tokenizer._backend == "tokie"
    except ImportError as e:
        pytest.skip(f"Could not import tokenizer for {model_name}: {e}")
    except Exception as e:
        if "not found in model".casefold() in str(e).casefold():
            pytest.skip(f"Skipping test with {model_name}. Backend not available")
        else:
            raise e


@pytest.mark.parametrize(
    "backend_str",
    ["hf_tokenizer", "tiktoken_tokenizer", "transformers_tokenizer"],
)
def test_encode_decode(request: pytest.FixtureRequest, backend_str: str, sample_text: str) -> None:
    """Test encoding and decoding with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = AutoTokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {e}")

    # Encode, Decode and Compare
    tokens = tokenizer.encode(sample_text)
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)

    if tokenizer._backend != "callable":
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert decoded == sample_text


@pytest.mark.parametrize("model_name", ["gpt2", "p50k_base"])
def test_string_init_encode_decode(model_name: str) -> None:
    """Test basic functionality of string initialized models."""
    try:
        tokenizer = AutoTokenizer(model_name)
        assert tokenizer is not None
        assert tokenizer._backend == "tokie"
        test_string = "Testing tokenizer_string_init_basic for Chonkie Tokenizers."
        tokens = tokenizer.encode(test_string)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)

        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        # Check if decoded strings preserves original words
        for word in [
            "testing",
            "Chonkie",
            "Tokenizers",
        ]:
            assert word.lower() in decoded.lower()
    except ImportError as e:
        pytest.skip(f"Skipping test. Could not import tokenizer for {model_name}: {e}")
    except Exception as e:
        if "not found in model".casefold() in str(e).casefold():
            pytest.skip(f"Skipping test with {model_name}. Backend not available")
        else:
            raise e


@pytest.mark.parametrize(
    "backend_str",
    [
        "hf_tokenizer",
        "tiktoken_tokenizer",
        "transformers_tokenizer",
        "callable_tokenizer",
    ],
)
def test_token_counting(
    request: pytest.FixtureRequest,
    backend_str: str,
    sample_text: str,
) -> None:
    """Test token counting with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = AutoTokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {e}")

    count = tokenizer.count_tokens(sample_text)
    assert isinstance(count, int)
    assert count > 0

    # Verify count matches encoded length
    if tokenizer._backend != "callable":
        assert count == len(tokenizer.encode(sample_text))


@pytest.mark.parametrize(
    "backend_str",
    ["hf_tokenizer", "tiktoken_tokenizer", "transformers_tokenizer"],
)
def test_batch_encode_decode(
    request: pytest.FixtureRequest,
    backend_str: str,
    sample_text_list: list[str],
) -> None:
    """Test batch encoding and decoding with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = AutoTokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {e}")

    batch_encoded = tokenizer.encode_batch(sample_text_list)
    assert isinstance(batch_encoded, list)
    assert len(batch_encoded) == len(sample_text_list)
    assert all(isinstance(tokens, list) for tokens in batch_encoded)
    assert all(len(tokens) > 0 for tokens in batch_encoded)
    assert all(all(isinstance(token, int) for token in tokens) for tokens in batch_encoded)

    if tokenizer._backend != "callable":
        batch_decoded = tokenizer.decode_batch(batch_encoded)
        assert isinstance(batch_decoded, list)
        assert len(batch_decoded) == len(sample_text_list)
        assert all(isinstance(text, str) for text in batch_decoded)
        assert batch_decoded == sample_text_list


@pytest.mark.parametrize(
    "backend_str",
    ["hf_tokenizer", "tiktoken_tokenizer", "transformers_tokenizer"],
)
def test_batch_counting(
    request: pytest.FixtureRequest,
    backend_str: str,
    sample_text_list: list[str],
) -> None:
    """Test batch token counting with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = AutoTokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {e}")

    # Test batch token count
    counts = tokenizer.count_tokens_batch(sample_text_list)
    assert isinstance(counts, list)
    assert len(counts) == len(sample_text_list)
    assert all(isinstance(c, int) for c in counts)
    assert all(c > 0 for c in counts)

    # Verify counts match encoded lengths
    if tokenizer._backend != "callable":
        encoded_lengths = [len(tokens) for tokens in tokenizer.encode_batch(sample_text_list)]
        assert counts == encoded_lengths


def test_tokenizer_raises_error_with_invalid_tokenizer() -> None:
    """Test if AutoTokenizer raises ValueError when initialized with an invalid tokenizer."""
    with pytest.raises(ValueError):
        AutoTokenizer(object())


def test_raises_correct_error() -> None:
    """Test if tokenizers raise expected errors."""
    tokenizer = AutoTokenizer(lambda x: len(x))

    assert tokenizer.count_tokens("test") == 4

    with pytest.raises(NotImplementedError):
        tokenizer.encode(
            "Ratatouille or Wall-E? Tell us which is the best Pixar movie on Discord.",
        )

    with pytest.raises(NotImplementedError):
        tokenizer.decode([0, 1, 2])

    with pytest.raises(NotImplementedError):
        tokenizer.encode_batch(["I", "Like", "Ratatouille", "Personally"])


### WordTokenizer Tests ###
def test_word_tokenizer_init(word_tokenizer: WordTokenizer) -> None:
    """Test WordTokenizer initialization."""
    assert word_tokenizer.vocab == [" "]
    assert len(word_tokenizer.token2id) == 1
    assert word_tokenizer.token2id[" "] == 0


def test_word_tokenizer_encode_decode(word_tokenizer: WordTokenizer, sample_text: str) -> None:
    """Test WordTokenizer encoding and decoding."""
    tokens = word_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)

    decoded = word_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded.strip() == sample_text.strip()


def test_word_tokenizer_batch_encode_decode(
    word_tokenizer: WordTokenizer,
    sample_text_list: list[str],
) -> None:
    """Test batch encode and decode with WordTokenizer."""
    encoded_batch = word_tokenizer.encode_batch(sample_text_list)
    assert isinstance(encoded_batch, list)
    assert all(isinstance(tokens, list) for tokens in encoded_batch)

    decoded_batch = word_tokenizer.decode_batch(encoded_batch)
    assert isinstance(decoded_batch, list)
    assert all(isinstance(text, str) for text in decoded_batch)
    for decoded_text, original_text in zip(decoded_batch, sample_text_list):
        assert decoded_text.strip() == original_text.strip()


def test_word_tokenizer_vocab_appends_new_words(
    word_tokenizer: WordTokenizer,
) -> None:
    """Test WordTokenizer appends new words to the vocabulary."""
    initial_vocab_size = len(word_tokenizer.vocab)
    test_str = "every tech bro should watch wall-e"
    word_tokenizer.encode(test_str)
    assert len(word_tokenizer.vocab) > initial_vocab_size
    for word in test_str.split():
        assert word in word_tokenizer.vocab


def test_word_tokenizer_repr() -> None:
    """Test string representation of tokenizers."""
    word_tokenizer = WordTokenizer()
    assert str(word_tokenizer) == "WordTokenizer(vocab_size=1)"


def test_word_tokenizer_multiple_encodings(
    word_tokenizer: WordTokenizer,
) -> None:
    """Test that vocabulary changes as expected over multiple encodings."""
    str1 = "Wall-E is truly a masterpiece that should be required viewing."
    str2 = "Ratatouille is truly a delightful film that every kid should watch."

    # Test WordTokenizer
    word_tokenizer.encode(str1)
    vocab_size1 = len(word_tokenizer.get_vocab())
    word_tokenizer.encode(str2)
    vocab_size2 = len(word_tokenizer.get_vocab())

    assert vocab_size2 > vocab_size1
    assert "Wall-E" in word_tokenizer.get_vocab()
    assert "Ratatouille" in word_tokenizer.get_vocab()
    assert word_tokenizer.get_token2id()["truly"] == word_tokenizer.encode("truly")[0]


### CharacterTokenizer Tests ###
def test_character_tokenizer_init(
    character_tokenizer: CharacterTokenizer,
) -> None:
    """Test CharacterTokenizer initialization."""
    assert character_tokenizer.vocab == [" "]
    assert len(character_tokenizer.token2id) == 1
    assert character_tokenizer.token2id[" "] == 0


def test_character_tokenizer_encode_decode(
    character_tokenizer: CharacterTokenizer,
    sample_text: str,
) -> None:
    """Test encoding and decoding with CharacterTokenizer."""
    tokens = character_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
    assert len(tokens) == len(sample_text)

    decoded = character_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded == sample_text


def test_character_tokenizer_count_tokens(
    character_tokenizer: CharacterTokenizer,
    sample_text: str,
    sample_text_list: list[str],
) -> None:
    """Test token counting with CharacterTokenizer."""
    count = character_tokenizer.count_tokens(sample_text)
    assert count == len(sample_text)


def test_character_tokenizer_batch_encode_decode(
    character_tokenizer: CharacterTokenizer,
    sample_text_list: list[str],
) -> None:
    """Test batch encoding and decoding with CharacterTokenizer."""
    batch_encoded = character_tokenizer.encode_batch(sample_text_list)
    assert isinstance(batch_encoded, list)
    assert all(isinstance(tokens, list) for tokens in batch_encoded)
    assert all(len(tokens) == len(text) for tokens, text in zip(batch_encoded, sample_text_list))

    batch_decoded = character_tokenizer.decode_batch(batch_encoded)
    assert isinstance(batch_decoded, list)
    assert all(isinstance(text, str) for text in batch_decoded)
    assert batch_decoded == sample_text_list


def test_character_tokenizer_count_tokens_batch(
    character_tokenizer: CharacterTokenizer,
    sample_text_list: list[str],
) -> None:
    """Test batch token counting with CharacterTokenizer."""
    counts = character_tokenizer.count_tokens_batch(sample_text_list)
    assert counts == [len(text) for text in sample_text_list]


def test_character_tokenizer_repr() -> None:
    """Test string representation of tokenizers."""
    character_tokenizer = CharacterTokenizer()
    assert str(character_tokenizer) == "CharacterTokenizer(vocab_size=1)"


def test_character_tokenizer_vocab_and_mapping(
    character_tokenizer: CharacterTokenizer,
    sample_text: str,
) -> None:
    """Test vocabulary evolution in CharacterTokenizer."""
    # Initial state
    assert character_tokenizer.get_vocab() == [" "]
    assert dict(character_tokenizer.get_token2id()) == {" ": 0}

    character_tokenizer.encode(sample_text)

    # Encoding text should add vocabulary
    # and update token2id mapping
    vocab = character_tokenizer.get_vocab()
    token2id = character_tokenizer.get_token2id()

    # Spot check vocabulary
    assert len(vocab) > 1

    assert isinstance(token2id, dict)
    assert all(isinstance(token, str) for token in token2id.keys())
    assert all(isinstance(idx, int) for idx in token2id.values())
    assert token2id[" "] == 0

    # Verify mapping consistency
    for token in vocab:
        assert token in token2id
        assert vocab[token2id[token]] == token

    for char in sample_text:
        assert char in vocab
        assert char in token2id


def test_character_tokenizer_multiple_encodings(
    character_tokenizer: CharacterTokenizer,
) -> None:
    """Test that vocabulary changes as expected over multiple encodings."""
    text1 = "Wall-E is truly a masterpiece that should be required viewing."
    text2 = "Ratatouille is truly a delightful film that every kid should watch."

    character_tokenizer.encode(text1)
    vocab_size1 = len(character_tokenizer.get_vocab())
    character_tokenizer.encode(text2)
    vocab_size2 = len(character_tokenizer.get_vocab())

    assert vocab_size2 > vocab_size1
    assert "u" in character_tokenizer.get_vocab()
    assert character_tokenizer.get_token2id()["u"] == character_tokenizer.encode("u")[0]


### ByteTokenizer Tests ###
def test_byte_tokenizer_init(byte_tokenizer: ByteTokenizer) -> None:
    """Test ByteTokenizer initialization."""
    assert byte_tokenizer.vocab == [" "]
    assert len(byte_tokenizer.token2id) == 1
    assert byte_tokenizer.token2id[" "] == 0


def test_byte_tokenizer_encode_decode(byte_tokenizer: ByteTokenizer, sample_text: str) -> None:
    """Test encoding and decoding with ByteTokenizer."""
    tokens = byte_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
    assert all(0 <= token <= 255 for token in tokens)

    decoded = byte_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded == sample_text


def test_byte_tokenizer_unicode_support(byte_tokenizer: ByteTokenizer) -> None:
    """Test ByteTokenizer with unicode characters."""
    unicode_text = "Hello, 世界! 🌍 Café"
    tokens = byte_tokenizer.encode(unicode_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)

    decoded = byte_tokenizer.decode(tokens)
    assert decoded == unicode_text


def test_byte_tokenizer_count_tokens(
    byte_tokenizer: ByteTokenizer,
    sample_text: str,
) -> None:
    """Test token counting with ByteTokenizer."""
    count = byte_tokenizer.count_tokens(sample_text)
    assert count == len(sample_text.encode("utf-8"))
    assert count == len(byte_tokenizer.encode(sample_text))


def test_byte_tokenizer_batch_encode_decode(
    byte_tokenizer: ByteTokenizer,
    sample_text_list: list[str],
) -> None:
    """Test batch encoding and decoding with ByteTokenizer."""
    batch_encoded = byte_tokenizer.encode_batch(sample_text_list)
    assert isinstance(batch_encoded, list)
    assert all(isinstance(tokens, list) for tokens in batch_encoded)
    assert all(
        all(isinstance(token, int) and 0 <= token <= 255 for token in tokens)
        for tokens in batch_encoded
    )

    batch_decoded = byte_tokenizer.decode_batch(batch_encoded)
    assert isinstance(batch_decoded, list)
    assert all(isinstance(text, str) for text in batch_decoded)
    assert batch_decoded == sample_text_list


def test_byte_tokenizer_count_tokens_batch(
    byte_tokenizer: ByteTokenizer,
    sample_text_list: list[str],
) -> None:
    """Test batch token counting with ByteTokenizer."""
    counts = byte_tokenizer.count_tokens_batch(sample_text_list)
    expected_counts = [len(text.encode("utf-8")) for text in sample_text_list]
    assert counts == expected_counts


def test_byte_tokenizer_repr() -> None:
    """Test string representation of ByteTokenizer."""
    byte_tokenizer = ByteTokenizer()
    assert str(byte_tokenizer) == "ByteTokenizer(vocab_size=1)"


def test_byte_tokenizer_empty_text(byte_tokenizer: ByteTokenizer) -> None:
    """Test ByteTokenizer with empty text."""
    assert byte_tokenizer.encode("") == []
    assert byte_tokenizer.decode([]) == ""
    assert byte_tokenizer.count_tokens("") == 0


def test_byte_tokenizer_decode_invalid_bytes() -> None:
    """Test ByteTokenizer error handling for invalid UTF-8 byte sequences."""
    byte_tokenizer = ByteTokenizer()

    # Invalid UTF-8 sequence
    invalid_bytes = [0xFF, 0xFE, 0xFD]
    with pytest.raises(ValueError, match="Decoding failed"):
        byte_tokenizer.decode(invalid_bytes)


def test_byte_tokenizer_ascii_vs_unicode() -> None:
    """Test byte count difference between ASCII and unicode."""
    byte_tokenizer = ByteTokenizer()

    ascii_text = "Hello"
    unicode_text = "世界"

    ascii_count = byte_tokenizer.count_tokens(ascii_text)
    unicode_count = byte_tokenizer.count_tokens(unicode_text)

    # ASCII: 5 characters = 5 bytes
    assert ascii_count == 5
    # Chinese characters: 2 characters = 6 bytes in UTF-8
    assert unicode_count == 6


def test_byte_tokenizer_with_autotokenizer() -> None:
    """Test ByteTokenizer initialization with AutoTokenizer."""
    tokenizer = AutoTokenizer("byte")
    assert isinstance(tokenizer.tokenizer, ByteTokenizer)
    assert tokenizer._backend == "chonkie"

    text = "Hello, 世界!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


### Edge Cases and Error Handling Tests ###


def test_tokenizer_empty_text() -> None:
    """Test tokenizer behavior with empty text."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    # Test empty string encoding
    assert char_tokenizer.encode("") == []
    # Word tokenizer returns [1] for empty string due to split behavior creating [""]
    word_encoded = word_tokenizer.encode("")
    assert len(word_encoded) == 1  # Contains one empty token

    # Test empty string token counting
    assert char_tokenizer.count_tokens("") == 0
    assert word_tokenizer.count_tokens("") == 1  # Empty string splits to one empty token

    # Test empty string decoding
    assert char_tokenizer.decode([]) == ""
    assert word_tokenizer.decode(word_encoded) == ""  # Should decode back to empty string


def test_tokenizer_special_characters() -> None:
    """Test tokenizer behavior with special characters and unicode."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    special_text = "Hello! 😀 你好 🌍 Café naïve résumé"

    # Test encoding and decoding with special characters
    char_tokens = char_tokenizer.encode(special_text)
    char_decoded = char_tokenizer.decode(char_tokens)
    assert char_decoded == special_text

    word_tokens = word_tokenizer.encode(special_text)
    word_decoded = word_tokenizer.decode(word_tokens)
    assert word_decoded == special_text


def test_tokenizer_whitespace_handling() -> None:
    """Test tokenizer behavior with various whitespace scenarios."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    # Test multiple spaces
    text_with_spaces = "hello    world"
    char_tokens = char_tokenizer.encode(text_with_spaces)
    assert len(char_tokens) == len(text_with_spaces)
    assert char_tokenizer.decode(char_tokens) == text_with_spaces

    # Test tabs and newlines
    text_with_whitespace = "hello\tworld\ntest"
    char_tokens = char_tokenizer.encode(text_with_whitespace)
    assert char_tokenizer.decode(char_tokens) == text_with_whitespace

    # Test leading/trailing spaces
    text_padded = "  hello world  "
    word_tokens = word_tokenizer.encode(text_padded)
    assert word_tokenizer.decode(word_tokens) == text_padded


def test_character_tokenizer_decode_invalid_tokens() -> None:
    """Test character tokenizer error handling for invalid tokens."""
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.encode("hello")  # Build some vocab

    # Test decoding with invalid token IDs
    with pytest.raises(ValueError, match="Decoding failed"):
        char_tokenizer.decode([999, 1000])  # Non-existent token IDs


def test_word_tokenizer_decode_invalid_tokens() -> None:
    """Test word tokenizer error handling for invalid tokens."""
    word_tokenizer = WordTokenizer()
    word_tokenizer.encode("hello world")  # Build some vocab

    # Test decoding with invalid token IDs
    with pytest.raises(ValueError, match="Decoding failed"):
        word_tokenizer.decode([999, 1000])  # Non-existent token IDs


def test_tokenizer_consistency_across_operations() -> None:
    """Test that encode/decode operations are consistent."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    test_text = "The quick brown fox jumps over the lazy dog."

    # Test character tokenizer consistency
    char_tokens = char_tokenizer.encode(test_text)
    char_count_direct = char_tokenizer.count_tokens(test_text)
    char_count_from_encode = len(char_tokens)
    assert char_count_direct == char_count_from_encode

    # Test word tokenizer consistency
    word_tokens = word_tokenizer.encode(test_text)
    word_count_direct = word_tokenizer.count_tokens(test_text)
    word_count_from_encode = len(word_tokens)
    assert word_count_direct == word_count_from_encode


def test_tokenizer_vocab_persistence() -> None:
    """Test that vocabulary persists across multiple operations."""
    char_tokenizer = CharacterTokenizer()

    # Encode first text
    text1 = "hello"
    char_tokenizer.encode(text1)
    vocab_after_first = len(char_tokenizer.get_vocab())

    # Encode same text again - vocab should not grow
    char_tokenizer.encode(text1)
    vocab_after_repeat = len(char_tokenizer.get_vocab())
    assert vocab_after_first == vocab_after_repeat

    # Encode new text - vocab should grow
    text2 = "xyz"  # New characters
    char_tokenizer.encode(text2)
    vocab_after_new = len(char_tokenizer.get_vocab())
    assert vocab_after_new > vocab_after_repeat


def test_word_tokenizer_single_character_words() -> None:
    """Test word tokenizer with single character words."""
    word_tokenizer = WordTokenizer()

    text = "I a m t e s t i n g"
    tokens = word_tokenizer.encode(text)
    decoded = word_tokenizer.decode(tokens)
    assert decoded == text

    # Check that single characters are treated as separate words
    assert word_tokenizer.count_tokens(text) == len(text.split(" "))


def test_tokenizer_large_text() -> None:
    """Test tokenizer performance with larger text."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    # Create a larger text by repeating
    base_text = "The quick brown fox jumps over the lazy dog. "
    large_text = base_text * 100  # 4300+ characters

    # Test character tokenizer
    char_tokens = char_tokenizer.encode(large_text)
    assert len(char_tokens) == len(large_text)
    char_decoded = char_tokenizer.decode(char_tokens)
    assert char_decoded == large_text

    # Test word tokenizer
    word_tokens = word_tokenizer.encode(large_text)
    word_decoded = word_tokenizer.decode(word_tokens)
    assert word_decoded == large_text


def test_tokenizer_numeric_content() -> None:
    """Test tokenizer behavior with numeric content."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    numeric_text = "123 456.789 -10 +20 1.23e-4"

    # Test character tokenizer with numbers
    char_tokens = char_tokenizer.encode(numeric_text)
    char_decoded = char_tokenizer.decode(char_tokens)
    assert char_decoded == numeric_text

    # Test word tokenizer with numbers
    word_tokens = word_tokenizer.encode(numeric_text)
    word_decoded = word_tokenizer.decode(word_tokens)
    assert word_decoded == numeric_text


### Additional Unified Tokenizer Tests ###


def test_tokenizer_backend_detection_accuracy() -> None:
    """Test that backend detection is accurate for different tokenizer types."""
    # Test character tokenizer backend detection
    char_tokenizer = AutoTokenizer(CharacterTokenizer())
    assert char_tokenizer._backend == "chonkie"

    # Test word tokenizer backend detection
    word_tokenizer = AutoTokenizer(WordTokenizer())
    assert word_tokenizer._backend == "chonkie"

    # Test byte tokenizer backend detection
    byte_tokenizer = AutoTokenizer(ByteTokenizer())
    assert byte_tokenizer._backend == "chonkie"


def test_tokenizer_with_non_standard_callable() -> None:
    """Test tokenizer with various callable types."""
    # Test with lambda
    lambda_tokenizer = AutoTokenizer(lambda x: len(x.split()))
    assert lambda_tokenizer._backend == "callable"
    assert lambda_tokenizer.count_tokens("hello world") == 2

    # Test with class method
    class CustomTokenizer:
        def __call__(self, text: str) -> int:
            return len(text.split(","))

    custom_tokenizer = AutoTokenizer(CustomTokenizer())
    assert custom_tokenizer._backend == "callable"
    assert custom_tokenizer.count_tokens("a,b,c") == 3


def test_tokenizer_initialization_edge_cases() -> None:
    """Test tokenizer initialization with edge cases."""
    # Test initialization with character string
    char_tokenizer = AutoTokenizer("character")
    assert isinstance(char_tokenizer.tokenizer, CharacterTokenizer)

    # Test initialization with word string
    word_tokenizer = AutoTokenizer("word")
    assert isinstance(word_tokenizer.tokenizer, WordTokenizer)

    # Test initialization with byte string
    byte_tokenizer = AutoTokenizer("byte")
    assert isinstance(byte_tokenizer.tokenizer, ByteTokenizer)


def test_tokenizer_batch_operations_consistency() -> None:
    """Test that batch operations are consistent with single operations."""
    try:
        tokenizer = AutoTokenizer("gpt2")
    except Exception:
        pytest.skip("GPT-2 tokenizer not available")

    texts = ["hello", "world", "test"]

    # Test encode batch consistency
    batch_encoded = tokenizer.encode_batch(texts)
    single_encoded = [tokenizer.encode(text) for text in texts]
    assert batch_encoded == single_encoded

    # Test decode batch consistency
    if tokenizer._backend != "callable":
        batch_decoded = tokenizer.decode_batch(batch_encoded)
        single_decoded = [tokenizer.decode(tokens) for tokens in batch_encoded]
        assert batch_decoded == single_decoded

    # Test count batch consistency
    batch_counts = tokenizer.count_tokens_batch(texts)
    single_counts = [tokenizer.count_tokens(text) for text in texts]
    assert batch_counts == single_counts


### Tokie Backend Tests ###


@pytest.mark.parametrize("model_name", ["gpt2", "p50k_base"])
def test_tokie_backend_string_init(model_name: str) -> None:
    """Test that string-loaded tokenizers use tokie backend."""
    tokenizer = AutoTokenizer(model_name)
    assert tokenizer._backend == "tokie"

    text = "Hello, world! Testing tokie backend."
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert "Hello" in decoded


def test_tokie_backend_batch_operations() -> None:
    """Test batch operations with tokie backend."""
    tokenizer = AutoTokenizer("gpt2")
    assert tokenizer._backend == "tokie"

    texts = ["hello world", "testing batch", "tokie is fast"]

    encoded = tokenizer.encode_batch(texts)
    assert len(encoded) == len(texts)
    assert all(len(e) > 0 for e in encoded)

    decoded = tokenizer.decode_batch(encoded)
    assert len(decoded) == len(texts)
    assert all(isinstance(d, str) for d in decoded)

    counts = tokenizer.count_tokens_batch(texts)
    assert len(counts) == len(texts)
    assert all(c > 0 for c in counts)
    assert counts == [len(e) for e in encoded]


def test_tokie_backend_instance_passthrough() -> None:
    """Test that passing a tokie instance directly works."""
    tokie_tok = TokieTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer(tokie_tok)
    assert tokenizer._backend == "tokie"

    text = "Instance passthrough test"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert "Instance" in decoded


def test_tokenizer_error_propagation() -> None:
    """Test that errors are properly propagated from underlying tokenizers."""
    char_tokenizer = AutoTokenizer(CharacterTokenizer())

    # Test that decoding invalid tokens raises appropriate error
    with pytest.raises(ValueError):
        char_tokenizer.decode([999, 1000])


@pytest.mark.parametrize("invalid_input", [None, 123, [], {}])
def test_tokenizer_invalid_initialization(invalid_input: Any) -> None:
    """Test tokenizer initialization with invalid inputs."""
    with pytest.raises(ValueError):
        AutoTokenizer(invalid_input)


### Additional Coverage Tests ###


def test_tokenizer_decode_batch_callable_error() -> None:
    """Test that decode_batch raises NotImplementedError for callable tokenizers."""
    callable_tokenizer = AutoTokenizer(lambda x: len(x.split()))

    with pytest.raises(NotImplementedError, match="Batch decoding not implemented"):
        callable_tokenizer.decode_batch([[1, 2], [3, 4]])


def test_tokenizer_encode_batch_callable_error() -> None:
    """Test that encode_batch raises NotImplementedError for callable tokenizers."""
    callable_tokenizer = AutoTokenizer(lambda x: len(x.split()))

    with pytest.raises(NotImplementedError, match="Batch encoding not implemented"):
        callable_tokenizer.encode_batch(["hello world", "test"])


def test_base_tokenizer_abstract_methods() -> None:
    """Test that BaseTokenizer cannot be instantiated with missing abstract methods."""
    from chonkie.tokenizer import Tokenizer as BaseTokenizer

    # Create a class that doesn't implement abstract methods
    class IncompleteTokenizer(BaseTokenizer):
        def __repr__(self) -> str:
            return "IncompleteTokenizer"

        # Missing: encode, decode, count_tokens implementations

    # This should raise TypeError because abstract methods aren't implemented
    with pytest.raises(TypeError):
        IncompleteTokenizer()


def test_base_tokenizer_not_implemented_errors() -> None:
    """Test BaseTokenizer raises NotImplementedError for abstract methods."""
    from chonkie.tokenizer import Tokenizer as BaseTokenizer

    # Create a partial implementation that only implements __repr__
    class PartialTokenizer(BaseTokenizer):
        def __repr__(self) -> str:
            return "PartialTokenizer"

        # Override abstract methods to call super() to trigger NotImplementedError
        def encode(self, text: str):
            return super().encode(text)

        def decode(self, tokens):
            return super().decode(tokens)

        def count_tokens(self, text: str):
            return super().count_tokens(text)

    # We can't instantiate this because it's still abstract, but we can test the error paths
    # by creating a fully concrete version that calls super()
    class TestTokenizer(BaseTokenizer):
        def __repr__(self) -> str:
            return "TestTokenizer"

        def tokenize(self, text: str):
            return super().tokenize(text)  # Should raise NotImplementedError

        def encode(self, text: str):
            return super().encode(text)  # Should raise NotImplementedError

        def decode(self, tokens):
            return super().decode(tokens)  # Should raise NotImplementedError

    tokenizer = TestTokenizer()

    # Test that each abstract method raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Tokenization not implemented"):
        tokenizer.tokenize("test")

    with pytest.raises(NotImplementedError, match="Encoding not implemented"):
        tokenizer.encode("test")

    with pytest.raises(NotImplementedError, match="Decoding not implemented"):
        tokenizer.decode([1, 2, 3])


def test_character_tokenizer_default_token2id() -> None:
    """Test the defaulttoken2id method of CharacterTokenizer."""
    char_tokenizer = CharacterTokenizer()

    # Test that the default token ID function works correctly
    initial_vocab_size = len(char_tokenizer.vocab)
    default_id = char_tokenizer.defaulttoken2id()
    assert default_id == initial_vocab_size

    # Add some characters and test again
    char_tokenizer.encode("abc")
    new_default_id = char_tokenizer.defaulttoken2id()
    assert new_default_id == len(char_tokenizer.vocab)


def test_word_tokenizer_tokenize_method() -> None:
    """Test the tokenize method of WordTokenizer directly."""
    word_tokenizer = WordTokenizer()

    # Test direct tokenize method
    text = "hello world test"
    tokens = word_tokenizer.tokenize(text)
    expected_tokens = text.split(" ")
    assert tokens == expected_tokens

    # Test with multiple spaces
    text_spaces = "hello  world   test"
    tokens_spaces = word_tokenizer.tokenize(text_spaces)
    expected_spaces = text_spaces.split(" ")
    assert tokens_spaces == expected_spaces


def test_tokenizer_transformers_batch_decode_path() -> None:
    """Test the transformers-specific batch decode path."""
    try:
        # Create a transformers tokenizer by passing an instance directly
        hf_tokenizer = HFAutoTokenizer.from_pretrained("gpt2")
        tokenizer = AutoTokenizer(hf_tokenizer)
        assert tokenizer._backend == "transformers"
        texts = ["hello", "world"]
        encoded = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(encoded)
        assert decoded == texts
    except Exception:
        pytest.skip("Transformers tokenizer not available")


def test_tokenizer_tiktoken_batch_operations() -> None:
    """Test tiktoken-specific batch operations."""
    try:
        import tiktoken

        tokenizer = AutoTokenizer(tiktoken.get_encoding("gpt2"))
        if tokenizer._backend == "tiktoken":
            texts = ["hello", "world"]

            # Test batch encode
            encoded = tokenizer.encode_batch(texts)
            assert len(encoded) == len(texts)

            # Test batch decode
            decoded = tokenizer.decode_batch(encoded)
            assert decoded == texts

            # Test batch count
            counts = tokenizer.count_tokens_batch(texts)
            assert len(counts) == len(texts)
        else:
            pytest.skip("Tiktoken backend not being used")
    except ImportError:
        pytest.skip("Tiktoken not available")


def test_tokenizer_tokenizers_batch_operations() -> None:
    """Test tokenizers-specific batch operations."""
    try:
        from tokenizers import Tokenizer as HFTokenizer

        hf_tokenizer = HFTokenizer.from_pretrained("gpt2")
        tokenizer = AutoTokenizer(hf_tokenizer)
        if tokenizer._backend == "tokenizers":
            texts = ["hello", "world"]

            # Test batch encode
            encoded = tokenizer.encode_batch(texts)
            assert len(encoded) == len(texts)

            # Test batch decode
            decoded = tokenizer.decode_batch(encoded)
            assert decoded == texts

            # Test batch count
            counts = tokenizer.count_tokens_batch(texts)
            assert len(counts) == len(texts)
        else:
            pytest.skip("Tokenizers backend not being used")
    except Exception:
        pytest.skip("HuggingFace tokenizers not available")


def test_tokenizer_chonkie_backend_paths() -> None:
    """Test chonkie-specific backend paths in unified tokenizer."""
    char_tokenizer = AutoTokenizer(CharacterTokenizer())
    word_tokenizer = AutoTokenizer(WordTokenizer())
    byte_tokenizer = AutoTokenizer(ByteTokenizer())

    # Test that chonkie backend is detected
    assert char_tokenizer._backend == "chonkie"
    assert word_tokenizer._backend == "chonkie"
    assert byte_tokenizer._backend == "chonkie"

    # Test chonkie-specific paths in methods
    text = "hello world"

    # Test encode path
    char_encoded = char_tokenizer.encode(text)
    word_encoded = word_tokenizer.encode(text)
    byte_encoded = byte_tokenizer.encode(text)
    assert len(char_encoded) == len(text)
    assert len(word_encoded) == len(text.split())
    assert len(byte_encoded) == len(text.encode("utf-8"))

    # Test count_tokens path
    char_count = char_tokenizer.count_tokens(text)
    word_count = word_tokenizer.count_tokens(text)
    byte_count = byte_tokenizer.count_tokens(text)
    assert char_count == len(text)
    assert word_count == len(text.split())
    assert byte_count == len(text.encode("utf-8"))

    # Test batch operations
    texts = ["hello", "world"]
    char_batch_encoded = char_tokenizer.encode_batch(texts)
    word_batch_encoded = word_tokenizer.encode_batch(texts)
    byte_batch_encoded = byte_tokenizer.encode_batch(texts)
    assert len(char_batch_encoded) == len(texts)
    assert len(word_batch_encoded) == len(texts)
    assert len(byte_batch_encoded) == len(texts)

    # Test batch count
    char_batch_counts = char_tokenizer.count_tokens_batch(texts)
    word_batch_counts = word_tokenizer.count_tokens_batch(texts)
    byte_batch_counts = byte_tokenizer.count_tokens_batch(texts)
    assert char_batch_counts == [len(text) for text in texts]
    assert word_batch_counts == [len(text.split()) for text in texts]
    assert byte_batch_counts == [len(text.encode("utf-8")) for text in texts]


def test_tokenizer_error_paths_comprehensive() -> None:
    """Test various error paths in tokenizer methods."""
    # Test invalid tokenizer creation with non-existent model
    with pytest.raises(ValueError, match="Tokenizer.+could not be loaded"):
        # This should try all backends and fail
        AutoTokenizer("definitely_not_a_real_model_name_12345_xyz")


def test_tokenizer_decode_batch_chonkie_path() -> None:
    """Test decode_batch specifically for chonkie backend."""
    char_tokenizer = AutoTokenizer(CharacterTokenizer())
    word_tokenizer = AutoTokenizer(WordTokenizer())
    byte_tokenizer = AutoTokenizer(ByteTokenizer())

    # Test chonkie backend decode_batch
    texts = ["hello", "world"]

    # Character tokenizer
    char_encoded = char_tokenizer.encode_batch(texts)
    char_decoded = char_tokenizer.decode_batch(char_encoded)
    assert char_decoded == texts

    # Word tokenizer
    word_encoded = word_tokenizer.encode_batch(texts)
    word_decoded = word_tokenizer.decode_batch(word_encoded)
    assert word_decoded == texts

    # Byte tokenizer
    byte_encoded = byte_tokenizer.encode_batch(texts)
    byte_decoded = byte_tokenizer.decode_batch(byte_encoded)
    assert byte_decoded == texts


def test_autotokenizer_wrapping() -> None:
    """Test that AutoTokenizer correctly unwraps when passed another AutoTokenizer."""
    # Create an AutoTokenizer
    tokenizer1 = AutoTokenizer("byte")
    assert tokenizer1._backend == "chonkie"
    assert isinstance(tokenizer1.tokenizer, ByteTokenizer)

    # Wrap it in another AutoTokenizer (this happens in chunkers)
    tokenizer2 = AutoTokenizer(tokenizer1)
    assert tokenizer2._backend == "chonkie"
    assert isinstance(tokenizer2.tokenizer, ByteTokenizer)

    # They should reference the same underlying tokenizer
    assert tokenizer2.tokenizer is tokenizer1.tokenizer

    # Test that it still works correctly
    text = "Hello, world!"
    assert tokenizer1.encode(text) == tokenizer2.encode(text)
    assert tokenizer1.count_tokens(text) == tokenizer2.count_tokens(text)


def test_tokenizer_base_repr_method() -> None:
    """Test the __repr__ method in BaseTokenizer."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()

    # Test that repr includes vocab size
    char_repr = repr(char_tokenizer)
    word_repr = repr(word_tokenizer)

    assert "CharacterTokenizer" in char_repr
    assert "WordTokenizer" in word_repr
    assert "vocab_size=" in char_repr
    assert "vocab_size=" in word_repr
