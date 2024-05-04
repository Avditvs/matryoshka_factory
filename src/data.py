"""Data loading for the matryoshka factory"""

import random
from typing import Iterable

from torch.utils.data import Dataset

from sentence_transformers import SentenceTransformer


class ParallelSentencesDataset(Dataset):
    def __init__(
        self,
        model: SentenceTransformer,
        sentences_pairs: Iterable[tuple[str, str]],
        inference_batch_size: int = 32,
    ):
        self.sentence_transformer = model
        self.inference_batch_size = inference_batch_size
        self.sentences_pairs = sentences_pairs
        self.cache = []

        random.shuffle(self.sentences_pairs)

    def __len__(self):
        return len(self.sentences_pairs)

    def generate_data(self):
        a_sentences = [a for a, _ in self.sentences_pairs]
        b_sentences = [b for _, b in self.sentences_pairs]

        a_embeddings = self.sentence_transformer.encode(
            a_sentences,
            batch_size=self.inference_batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        b_embeddings = self.sentence_transformer.encode(
            b_sentences,
            batch_size=self.inference_batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True
        )

        for a, b in zip(a_embeddings, b_embeddings):
            self.cache.append((a, b))

    def __getitem__(self, idx):
        if len(self.cache) == 0:
            self.generate_data()

        return self.sentences_pairs[idx], self.cache[idx]
