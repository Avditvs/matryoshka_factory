from abc import ABC
import datasets
from datasets import load_dataset
import random


class BaseDataset(ABC):
    def get_sentence_pairs(self):
        raise NotImplementedError


class STSBDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        languages: list[str] = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt",
            "ru",
            "zh",
        ],
        passage_prefix: str = "passage: ",
    ):
        self.split = split
        self.languages = languages

        ds = [
            load_dataset("stsb_multi_mt", c, split=self.split) for c in self.languages
        ]

        self.dataset = datasets.concatenate_datasets(ds)
        self.passage_prefix = passage_prefix

    def get_sentence_pairs(self):
        return [
            (
                self.passage_prefix + example["sentence1"],
                self.passage_prefix + example["sentence2"],
            )
            for example in self.dataset
        ]


class MrTyDiDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        languages: list[str] = [
            "combined",
        ],
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        self.split = split
        self.languages = languages
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        ds = [
            load_dataset("castorini/mr-tydi", c, split=self.split)
            for c in self.languages
        ]

        self.dataset = datasets.concatenate_datasets(ds)

    def get_sentence_pairs(
        self, limit_examples: int = 64, num_positives: int = 1, num_negatives: int = 1
    ):
        self.sentence_pairs = []
        for example in self.dataset:
            for positive_passage in example["positive_passages"][:limit_examples]:
                self.sentence_pairs.append(
                    (
                        self.query_prefix + example["query"],
                        self.passage_prefix + positive_passage["text"],
                    )
                )

            for negative_passage in example["negative_passages"][:limit_examples]:
                self.sentence_pairs.append(
                    (
                        self.query_prefix + example["query"],
                        self.passage_prefix + negative_passage["text"],
                    )
                )

            for i in range(min(len(example["positive_passages"]) // 2, num_positives)):
                self.sentence_pairs.append(
                    (
                        self.passage_prefix
                        + example["positive_passages"][2 * i]["text"],
                        self.passage_prefix
                        + example["positive_passages"][2 * i + 1]["text"],
                    )
                )

            for i in range(min(len(example["negative_passages"]) // 2, num_negatives)):
                self.sentence_pairs.append(
                    (
                        self.passage_prefix
                        + example["negative_passages"][2 * i]["text"],
                        self.passage_prefix
                        + example["negative_passages"][2 * i + 1]["text"],
                    )
                )

        return self.sentence_pairs


class QuoraDataset(BaseDataset):
    def __init__(self, split: str = "train"):
        self.split = split
        self.dataset = load_dataset("quora", split=self.split)
        self.query_prefix = "query: "

    def get_sentence_pairs(self, sample_rate: float = 1.0):
        return [
            (
                self.query_prefix + example["questions"]["text"][0],
                self.query_prefix + example["questions"]["text"][1],
            )
            for example in self.dataset
            if random.random() < sample_rate
        ]
