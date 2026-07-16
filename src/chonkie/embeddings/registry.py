"""Registry for embedding implementations with pattern matching support."""

import re
from typing import Any, Optional, Pattern, Union

from .azure_openai import AzureOpenAIEmbeddings
from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings
from .cloudflare import CloudflareEmbeddings
from .cohere import CohereEmbeddings
from .deepinfra import DeepInfraEmbeddings
from .gemini import GeminiEmbeddings
from .jina import JinaEmbeddings
from .litellm import LiteLLMEmbeddings
from .mistral import MistralEmbeddings
from .mixedbread import MixedbreadEmbeddings
from .model2vec import Model2VecEmbeddings
from .nomic import NomicEmbeddings
from .openai import OpenAIEmbeddings
from .sentence_transformer import SentenceTransformerEmbeddings
from .together import TogetherEmbeddings


class EmbeddingsRegistry:
    """Registry for embedding implementations with pattern matching support."""

    # Create a registry for the model names, provider aliases, patterns, and supported types
    model_registry: dict[str, type[BaseEmbeddings]] = {}
    provider_registry: dict[str, type[BaseEmbeddings]] = {}
    pattern_registry: dict[Pattern, type[BaseEmbeddings]] = {}
    type_registry: dict[str, type[BaseEmbeddings]] = {}

    @classmethod
    def register_model(cls, name: str, embedding_cls: type[BaseEmbeddings]) -> None:
        """Register a new embeddings implementation.

        Args:
            name: Unique identifier for this implementation
            embedding_cls: The embeddings class to register

        """
        if not issubclass(embedding_cls, BaseEmbeddings):
            raise ValueError(f"{embedding_cls} must be a subclass of BaseEmbeddings")

        cls.model_registry[name] = embedding_cls

    @classmethod
    def register_provider(
        cls,
        alias: str,
        embeddings_cls: type[BaseEmbeddings],
    ) -> None:
        """Register a new provider.

        Args:
            alias: Unique identifier for this provider
            embeddings_cls: The embeddings class to register

        """
        if not issubclass(embeddings_cls, BaseEmbeddings):
            raise ValueError(f"{embeddings_cls} must be a subclass of BaseEmbeddings")

        cls.provider_registry[alias] = embeddings_cls

    @classmethod
    def register_pattern(cls, pattern: str, embeddings_cls: type[BaseEmbeddings]) -> None:
        """Register a new pattern."""
        if not issubclass(embeddings_cls, BaseEmbeddings):
            raise ValueError(f"{embeddings_cls} must be a subclass of BaseEmbeddings")

        compiled_pattern = re.compile(pattern)
        cls.pattern_registry[compiled_pattern] = embeddings_cls

    @classmethod
    def register_types(
        cls,
        types: Union[str, list[str]],
        embeddings_cls: type[BaseEmbeddings],
    ) -> None:
        """Register a new type."""
        if not issubclass(embeddings_cls, BaseEmbeddings):
            raise ValueError(f"{embeddings_cls} must be a subclass of BaseEmbeddings")

        if isinstance(types, str):
            cls.type_registry[types] = embeddings_cls
        elif isinstance(types, list):
            for type in types:
                cls.type_registry[type] = embeddings_cls
        else:
            raise ValueError(f"Invalid types: {types}")

    @classmethod
    def get_provider(cls, alias: str) -> Optional[type[BaseEmbeddings]]:
        """Get the embeddings class for a given provider alias."""
        return cls.provider_registry.get(alias)

    @classmethod
    def match(cls, identifier: str) -> Optional[type[BaseEmbeddings]]:
        """Find matching embeddings class using both exact matches and patterns.

        Args:
            identifier: String to match against registry entries

        Returns:
            Matching embeddings class or None if no match found

        Examples:
            # Match exact name
            cls.match("sentence-transformers") -> SentenceTransformerEmbeddings

            # Match OpenAI pattern
            cls.match("openai://my-embedding") -> OpenAIEmbeddings

            # Match model name pattern
            cls.match("text-embedding-ada-002") -> OpenAIEmbeddings

        """
        # Firstly, we'll try to see if the provider alias is provided
        if "://" in identifier:
            provider, model_name = identifier.split("://", 1)
            if provider in cls.provider_registry:
                return cls.provider_registry[provider]

        # Now, let's try to get a match from the model name
        if identifier in cls.model_registry:
            return cls.model_registry[identifier]

        # We couldn't match the model name and there's no provider alias mentioned either.
        # Let's try to get a match from the pattern registry
        for pattern, embeddings_cls in cls.pattern_registry.items():
            if pattern.match(identifier):
                return embeddings_cls

        # We couldn't find a proper match for the model name or the provider alias.
        return None

    @classmethod
    def wrap(cls, object: Any, **kwargs: Any) -> BaseEmbeddings:
        """Wrap an object in the appropriate embeddings class.

        The objects that are handled here could be either a Model or Client object.

        Args:
            object: Name of the embeddings implementation
            **kwargs: Additional arguments passed to the embeddings constructor

        Returns:
            Initialized embeddings instance

        """
        # Check the object type and wrap it in the appropriate embeddings class
        if isinstance(object, BaseEmbeddings):
            return object
        elif isinstance(object, str):
            embeddings_cls = cls.match(object)
            if embeddings_cls is None:
                raise ValueError(f"No matching embeddings implementation found for: {object}")
            return embeddings_cls(object, **kwargs)  # ty: ignore[too-many-positional-arguments]
        else:
            # Loop through all the registered embeddings and check if the object is an instance of any of them
            for type_alias, embeddings_cls in cls.type_registry.items():
                if type_alias in str(type(object)):
                    return embeddings_cls(object, **kwargs)  # ty: ignore[too-many-positional-arguments]
        raise ValueError(f"Unsupported object type for embeddings: {object}")


# Register all the available embeddings in the EmbeddingsRegistry!
# This is essential for the `AutoEmbeddings` to work properly.

# Register SentenceTransformer embeddings with pattern
EmbeddingsRegistry.register_provider("st", SentenceTransformerEmbeddings)
EmbeddingsRegistry.register_pattern(
    r"^sentence-transformers/|^all-minilm-|^paraphrase-|^multi-qa-|^msmarco-",
    SentenceTransformerEmbeddings,
)
EmbeddingsRegistry.register_types(
    "SentenceTransformer",
    SentenceTransformerEmbeddings,
)

for model in [
    "all-minilm-l6-v2",
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-MiniLM-L12-v2",
    "all-distilroberta-v1",
    "all-roberta-large-v1",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "distiluse-base-multilingual-cased-v1",
    "distiluse-base-multilingual-cased-v2",
    "paraphrase-xlm-r-multilingual-v1",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-MiniLM-L3-v2",
    "paraphrase-mpnet-base-v2",
    "paraphrase-distilroberta-base-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-TinyBERT-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-distilbert-cos-v1",
    "multi-qa-mpnet-base-cos-v1",
    "msmarco-distilbert-base-v4",
    "msmarco-roberta-base-v3",
    "msmarco-distilbert-base-tas-b",
    "msmarco-MiniLM-L6-cos-v5",
    "msmarco-MiniLM-L12-cos-v5",
    "gtr-t5-large",
    "gtr-t5-xl",
    "sentence-t5-large",
    "sentence-t5-xl",
    "LaBSE",
    "allenai-specter",
    "average_word_embeddings_glove.6B.300d",
    "average_word_embeddings_komninos",
]:
    EmbeddingsRegistry.register_model(model, SentenceTransformerEmbeddings)

# Register OpenAI embeddings with pattern
EmbeddingsRegistry.register_provider("openai", OpenAIEmbeddings)
EmbeddingsRegistry.register_pattern(r"^text-embedding-", OpenAIEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-ada-002", OpenAIEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-3-small", OpenAIEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-3-large", OpenAIEmbeddings)

# Register Azure OpenAI embeddings
EmbeddingsRegistry.register_provider("azure_openai", AzureOpenAIEmbeddings)

# Register model2vec embeddings
EmbeddingsRegistry.register_provider("model2vec", Model2VecEmbeddings)
EmbeddingsRegistry.register_pattern(
    r"^minishlab/|^minishlab/potion-base-|^minishlab/potion-|^potion-",
    Model2VecEmbeddings,
)
EmbeddingsRegistry.register_types("Model2Vec", Model2VecEmbeddings)

# Register Cohere embeddings with pattern
EmbeddingsRegistry.register_provider("cohere", CohereEmbeddings)
EmbeddingsRegistry.register_pattern(r"^cohere|^embed-", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-light-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-multilingual-light-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-v2.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-light-v2.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-multilingual-v2.0", CohereEmbeddings)

# Register Jina embeddings
EmbeddingsRegistry.register_provider("jina", JinaEmbeddings)
EmbeddingsRegistry.register_pattern(r"^jina|^jinaai", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v3", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-en", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-es", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-de", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-zh", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-code", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v4", JinaEmbeddings)

# Register Voyage embeddings (via Catsu for better reliability and features)
EmbeddingsRegistry.register_provider("voyageai", CatsuEmbeddings)
EmbeddingsRegistry.register_pattern(r"^voyage|^voyageai", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-3-large", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-3", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-3-lite", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-code-3", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-finance-2", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-law-2", CatsuEmbeddings)
EmbeddingsRegistry.register_model("voyage-code-2", CatsuEmbeddings)

# Register Gemini embeddings
EmbeddingsRegistry.register_provider("gemini", GeminiEmbeddings)
EmbeddingsRegistry.register_pattern(
    r"^text-embedding-004|^embedding-001|^gemini-embedding",
    GeminiEmbeddings,
)
EmbeddingsRegistry.register_model("text-embedding-004", GeminiEmbeddings)
EmbeddingsRegistry.register_model("embedding-001", GeminiEmbeddings)
EmbeddingsRegistry.register_model("gemini-embedding-exp-03-07", GeminiEmbeddings)

# Register Catsu embeddings (unified provider for 11+ APIs)
EmbeddingsRegistry.register_provider("catsu", CatsuEmbeddings)
# Catsu supports patterns from multiple providers
EmbeddingsRegistry.register_pattern(r"^mistral-embed", CatsuEmbeddings)
EmbeddingsRegistry.register_pattern(r"^nomic-embed", CatsuEmbeddings)
EmbeddingsRegistry.register_pattern(r"^@cf/", CatsuEmbeddings)
EmbeddingsRegistry.register_pattern(r"^mxbai-embed", CatsuEmbeddings)
# Register specific Catsu-supported models
EmbeddingsRegistry.register_model("mistral-embed", CatsuEmbeddings)
EmbeddingsRegistry.register_model("nomic-embed-text-v1.5", CatsuEmbeddings)
EmbeddingsRegistry.register_model("mxbai-embed-large-v1", CatsuEmbeddings)

# Register Mistral embeddings
EmbeddingsRegistry.register_provider("mistral", MistralEmbeddings)

# Register Together embeddings
EmbeddingsRegistry.register_provider("together", TogetherEmbeddings)

# Register Mixedbread embeddings
EmbeddingsRegistry.register_provider("mixedbread", MixedbreadEmbeddings)

# Register Nomic embeddings
EmbeddingsRegistry.register_provider("nomic", NomicEmbeddings)

# Register DeepInfra embeddings
EmbeddingsRegistry.register_provider("deepinfra", DeepInfraEmbeddings)

# Register Cloudflare embeddings
EmbeddingsRegistry.register_provider("cloudflare", CloudflareEmbeddings)

# Register LiteLLM embeddings
EmbeddingsRegistry.register_provider("litellm", LiteLLMEmbeddings)
