from sentence_transformers import SentenceTransformer

from components.data import ParallelSentencesDataset
from components.finetuning import MatryoshkaTrainer, TrainingArguments

from datasets_classes import STSBDataset, MrTyDiDataset, QuoraDataset

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
stsbd_pairs = stsbd.get_sentence_pairs()
print(f"Number of STSB pairs: {len(stsbd_pairs)}")
sentences_pairs.extend(stsbd_pairs)
mrtydi = MrTyDiDataset()
mrtydi_pairs = mrtydi.get_sentence_pairs(
    limit_examples=3, num_negatives=2, num_positives=2
)
print(f"Number of MrTyDi pairs: {len(mrtydi_pairs)}")
sentences_pairs.extend(mrtydi_pairs)
quora = QuoraDataset()
quora_pairs = quora.get_sentence_pairs(sample_rate=0.5)
print(f"Number of Quora pairs: {len(quora_pairs)}")
sentences_pairs.extend(quora_pairs)

print(f"Total number of sentence pairs: {len(sentences_pairs)}")


parallel_dataset = ParallelSentencesDataset(
    teacher_model, sentences_pairs, inference_batch_size=64
)

iteration = 0

args = TrainingArguments(
    save_steps=100,
    per_device_batch_size=512,
    gradient_accumulation_steps=1,
    use_amp=True,
    lr=3e-5,
    model_save_path=f"models/{output_name}-matryoshka-datasets-{iteration}",
    matryoshka_dims=(384, 256, 192, 128, 64, 32),
    num_epochs=20,
    warmup_steps=256,
)

args.model_name = student_model_name

trainer = MatryoshkaTrainer(student_model, args)
trainer.fit(args, parallel_dataset)
