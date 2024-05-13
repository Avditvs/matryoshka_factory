from sentence_transformers import SentenceTransformer

from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset
from components.data import ParallelSentencesDataset
from components.finetuning import MatryoshkaTrainer, TrainingArguments


teacher_model_name = "intfloat/multilingual-e5-small"
teacher_model = SentenceTransformer(teacher_model_name)

student_model_name = "intfloat/multilingual-e5-small"
student_model = SentenceTransformer(student_model_name)

output_name = student_model_name.replace("/", "-")


configs = ['de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh']

ds = [load_dataset("stsb_multi_mt", c, split='train') for c in configs]

dataset = datasets.concatenate_datasets(ds)

sentences_pairs = [(example["sentence1"], example["sentence2"]) for example in dataset]
parallel_dataset = ParallelSentencesDataset(teacher_model, sentences_pairs, inference_batch_size=32)

train_dataloader = DataLoader(parallel_dataset, shuffle=True, batch_size=256)

iteration = 3

args = TrainingArguments(
    train_dataloader=train_dataloader,
    save_steps=100,
    train_batch_size=256,
    use_amp=True,
    lr=1e-5,
    model_save_path = f"{output_name}-matryoshka-passage-{iteration}",
    matryoshka_dims = (384, 256, 192, 128, 64, 32, 16, 8),
    num_epochs=20,
    warmup_steps=256,
)

args.model_name = student_model_name

trainer = MatryoshkaTrainer(student_model, args)
trainer.fit(args)
