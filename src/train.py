from sentence_transformers import SentenceTransformer

from torch.utils.data import DataLoader
from datasets import load_dataset
from components.data import ParallelSentencesDataset
from components.finetuning import MatryoshkaTrainer, TrainingArguments


teacher_model_name = "intfloat/multilingual-e5-small"
teacher_model = SentenceTransformer(teacher_model_name)

student_model_name = "intfloat/multilingual-e5-small"
student_model = SentenceTransformer(student_model_name)


dataset = load_dataset("stsb_multi_mt", name="en", split="train")
sentences_pairs = [(example["sentence1"], example["sentence2"]) for example in dataset]
parallel_dataset = ParallelSentencesDataset(teacher_model, sentences_pairs, inference_batch_size=32)

train_dataloader = DataLoader(parallel_dataset, shuffle=True, batch_size=32)

args = TrainingArguments(
    train_dataloader=train_dataloader,
    save_steps=100
)

trainer = MatryoshkaTrainer(student_model, args)
trainer.fit(args)
