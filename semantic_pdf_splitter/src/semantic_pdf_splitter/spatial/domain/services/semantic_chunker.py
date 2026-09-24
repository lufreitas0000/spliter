from typing import List
import numpy as np

from semantic_pdf_splitter.spatial.domain.models import TextSegment, SemanticChunk
from semantic_pdf_splitter.spatial.domain.ports import EmbeddingProvider

class SemanticChunker:
    def __init__(self, embedding_provider: EmbeddingProvider, similarity_threshold: float = 0.75):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

    def chunk_segments(self, segments: List[TextSegment]) -> List[SemanticChunk]:
        if not segments:
            return []

        texts = [segment.text for segment in segments]
        embeddings = self.embedding_provider.generate_embeddings(texts)

        for i, segment in enumerate(segments):
            segment.embedding = embeddings[i]

        chunks = []
        current_chunk_segments = [segments[0]]

        for i in range(1, len(segments)):
            vec_u = np.array(segments[i-1].embedding)
            vec_v = np.array(segments[i].embedding)

            norm_u = np.linalg.norm(vec_u)
            norm_v = np.linalg.norm(vec_v)

            if norm_u == 0 or norm_v == 0:
                similarity = 0.0
            else:
                similarity = np.dot(vec_u, vec_v) / (norm_u * norm_v)

            if similarity >= self.similarity_threshold:
                current_chunk_segments.append(segments[i])
            else:
                # Calculate average embedding for the current chunk
                chunk_embeddings = [seg.embedding for seg in current_chunk_segments if seg.embedding is not None]
                avg_embedding = np.mean(chunk_embeddings, axis=0).tolist() if chunk_embeddings else []
                chunks.append(SemanticChunk(segments=current_chunk_segments, average_embedding=avg_embedding))
                current_chunk_segments = [segments[i]]

        # Process the last chunk
        if current_chunk_segments:
            chunk_embeddings = [seg.embedding for seg in current_chunk_segments if seg.embedding is not None]
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist() if chunk_embeddings else []
            chunks.append(SemanticChunk(segments=current_chunk_segments, average_embedding=avg_embedding))

        return chunks
