import pytest
from typing import List

from semantic_pdf_splitter.spatial.domain.models import TextSegment
from semantic_pdf_splitter.spatial.domain.ports import EmbeddingProvider
from semantic_pdf_splitter.spatial.domain.services.semantic_chunker import SemanticChunker

class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, predefined_embeddings: List[List[float]]):
        self.predefined_embeddings = predefined_embeddings
        self.call_count = 0

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        return self.predefined_embeddings[:len(texts)]

def test_semantic_chunker_groups_similar_segments():
    # Similar embeddings should group together (parallel vectors)
    # Different embeddings should split apart (orthogonal vectors)
    embeddings = [
        [1.0, 0.0, 0.0],  # A (similar to B)
        [1.0, 0.1, 0.0],  # B (similar to A)
        [0.0, 1.0, 0.0],  # C (orthogonal to B - should split)
        [0.0, 1.0, 0.1],  # D (similar to C)
    ]

    provider = MockEmbeddingProvider(embeddings)
    chunker = SemanticChunker(embedding_provider=provider, similarity_threshold=0.75)

    segments = [
        TextSegment(text="A", page_number=1, bounding_box=(0,0,10,10)),
        TextSegment(text="B", page_number=1, bounding_box=(0,10,10,20)),
        TextSegment(text="C", page_number=1, bounding_box=(0,20,10,30)),
        TextSegment(text="D", page_number=1, bounding_box=(0,30,10,40)),
    ]

    chunks = chunker.chunk_segments(segments)

    assert provider.call_count == 1
    assert len(chunks) == 2

    # Check Chunk 1
    assert len(chunks[0].segments) == 2
    assert chunks[0].segments[0].text == "A"
    assert chunks[0].segments[1].text == "B"
    # avg of [1,0,0] and [1,0.1,0] -> [1, 0.05, 0]
    assert chunks[0].average_embedding == [1.0, 0.05, 0.0]

    # Check Chunk 2
    assert len(chunks[1].segments) == 2
    assert chunks[1].segments[0].text == "C"
    assert chunks[1].segments[1].text == "D"
    assert chunks[1].average_embedding == [0.0, 1.0, 0.05]

def test_semantic_chunker_handles_empty_input():
    provider = MockEmbeddingProvider([])
    chunker = SemanticChunker(embedding_provider=provider)

    chunks = chunker.chunk_segments([])
    assert len(chunks) == 0

def test_semantic_chunker_handles_zero_vectors():
    embeddings = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    provider = MockEmbeddingProvider(embeddings)
    chunker = SemanticChunker(embedding_provider=provider, similarity_threshold=0.75)

    segments = [
        TextSegment(text="A", page_number=1, bounding_box=(0,0,10,10)),
        TextSegment(text="B", page_number=1, bounding_box=(0,10,10,20)),
    ]

    chunks = chunker.chunk_segments(segments)

    assert len(chunks) == 2  # Should split because similarity is 0.0 for zero vectors
    assert len(chunks[0].segments) == 1
    assert len(chunks[1].segments) == 1
