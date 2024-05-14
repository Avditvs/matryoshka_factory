from sentence_transformers import SentenceTransformer

from torch.utils.data import DataLoader

from components.data import ParallelSentencesDataset
from components.finetuning import MatryoshkaTrainer, TrainingArguments

from datasets_classes import STSBDataset, MrTyDiDataset

max_seq_len = 128

teacher_model_name = "intfloat/multilingual-e5-small"
teacher_model = SentenceTransformer(teacher_model_name)
teacher_model.max_seq_length = max_seq_len

student_model_name = "intfloat/multilingual-e5-small"
student_model = SentenceTransformer(student_model_name)
student_model.max_seq_length = max_seq_len

output_name = student_model_name.replace("/", "-")


sentences_pairs = []
stsbd = STSBDataset()
sentences_pairs.extend(stsbd.get_sentence_pairs())
mrtydi = MrTyDiDataset()
sentences_pairs.extend(mrtydi.get_sentence_pairs(limit_examples=1))


parallel_dataset = ParallelSentencesDataset(
    teacher_model, sentences_pairs, inference_batch_size=64
)

train_dataloader = DataLoader(parallel_dataset, shuffle=True, batch_size=512)

iteration = 0

args = TrainingArguments(
    train_dataloader=train_dataloader,
    save_steps=100,
    train_batch_size=512,
    use_amp=True,
    lr=3e-5,
    model_save_path=f"{output_name}-matryoshka-datasets-{iteration}",
    matryoshka_dims=(384, 256, 192, 128, 64, 32),
    num_epochs=20,
    warmup_steps=256,
)

args.model_name = student_model_name

trainer = MatryoshkaTrainer(student_model, args)
trainer.fit(args)
