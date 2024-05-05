"""Class for distilling a model into a matryoshka model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

from .matryoshka import MatryoshkaLoss


@dataclass
class TrainingArguments:
    train_dataloader: torch.utils.data.DataLoader
    train_loss: nn.Module = nn.CosineSimilarity()
    num_epochs: int = 10
    model_save_path: str = "matryoshka_model_2"
    warmup_steps: int = 100
    train_batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10
    save_steps: int = 1000
    matryoshka_dims: tuple[int] = (384, 256, 192, 128, 96, 64, 48, 32)
    use_amp: bool = False
    device_type: str = "cuda"


class MatryoshkaTrainer:
    def __init__(self, model: SentenceTransformer, training_args: TrainingArguments):
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = self.prepare_optimizer(training_args)
        self.matroyshka_loss = MatryoshkaLoss(
            training_args.train_loss, matryoshka_dims=training_args.matryoshka_dims
        )
        self.final_loss = nn.MSELoss()
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=len(training_args.train_dataloader)
            * training_args.num_epochs,
        )

        self.scaler = torch.GradScaler()

    def prepare_optimizer(
        self, training_args: TrainingArguments
    ) -> torch.optim.Optimizer:
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_args.lr,
            eps=training_args.adam_epsilon,
        )

    def fit(self, training_args: TrainingArguments):
        total_steps = 0
        for epoch in range(training_args.num_epochs):
            self.model.train()
            for step, (original_sentences, original_embeddings) in enumerate(
                training_args.train_dataloader
            ):
                self.optimizer.zero_grad()

                with torch.autocast(
                    training_args.device_type, enabled=training_args.use_amp
                ):

                    input_tokens = [
                        {k: v.to(self.model.device) for k, v in sub_batch.items()}
                        for sub_batch in original_sentences
                    ]
                    outputs_student = [
                        self.model(i)["sentence_embedding"] for i in input_tokens
                    ]

                    student_similarities = self.matroyshka_loss(
                        outputs_student, no_sum=True
                    )

                    teacher_similarities = [
                        training_args.train_loss(
                            original_embeddings[0], original_embeddings[1]
                        )
                    ] * len(student_similarities)

                    weights = self.matroyshka_loss.matryoshka_weights
                    l0 = self.final_loss(
                        student_similarities[0], teacher_similarities[0]
                    )
                    loss = l0 * weights[0]

                    losses = [l0]
                    for w, student_sim, teacher_sim in zip(
                        self.matroyshka_loss.matryoshka_weights[1:],
                        student_similarities[1:],
                        teacher_similarities[1:],
                    ):
                        l = self.final_loss(student_sim, teacher_sim)
                        loss += l * w
                        losses.append(l.item())

                    loss /= len(weights)

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training_args.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scheduler.step()

                self.scaler.update()

                if total_steps % training_args.save_steps == 0:
                    self.model.save(training_args.model_save_path)
                displayed_losses = {
                    d: l for d, l in zip(self.matroyshka_loss.matryoshka_dims, losses)
                }

                total_steps += 1

                print(
                    f"Epoch: {total_steps/len(training_args.train_dataloader)}, Step: {total_steps}, Loss: {loss.item()}, lr: {self.scheduler.get_last_lr()[0]}"
                )
                for k, v in displayed_losses.items():
                    # print with alined numeric anddecimal points
                    print(f"dim: {k:03.0f}: {v:.5f}")
