import argparse
import time
import uuid
from pathlib import Path
from typing import Any

import mlflow
import onnx
import torch
from datasets import load_dataset


class TrainingSession:

    train_loader: torch.utils.data.Dataloader
    train_iter: iter

    valid_loader: torch.utils.data.DataLoader
    valid_iter: iter

    def __init__(
        self, epochs=10, batch_size=32, lr=0.001, save_dir="checkpoints", device="cuda"
    ):
        self.model_information = {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir
        self.device = torch.device(device)
        # TODO: Init hyper parameters.

        train_dataset = load_dataset(
            "dataset.py",  # TODO: Add your dataset.py path here.
            split="train",
            streaming=True,
            trust_remote_code=True,
            config_kwargs={},  # TODO: Add your dataset config here.
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset.with_format("numpy"), batch_size=self.batch_size
        )

        self.train_iter = iter(self.train_loader)
        self.model = None  # TODO: Add your model here.
        self.criterion = None  # TODO: Set your criterion.
        self.optimizer = None  # TODO: Set your optimizer.
        self.scheduler = None  # TODO: Set your scheduler.

    def train_and_valid(self):
        if self.criterion is None or self.optimizer is None:
            raise ValueError(
                "You must define a loss function and an optimizer before training."
            )

        self.model.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader):.4f}"
            )

            # TODO: Add validation here.
            # If you are satisfied with the metrics you can then save the model with `save_model`.

        # After training on multiple epochs you can inspect self.model_information to decide the one that
        # produced the best results and call `export_model_mlflow`.

    def save_model(self, metrics: dict[str, Any]) -> None:
        """At the end of each epoch, if the metrics indicate that the model's performance
        has improved compared to the previous epoch, the model is saved in ONNX format in self.save_dir.
        """
        save_path = Path(
            self.save_dir,
            f"model_{str(uuid.uuid4())[:8]}_{int(time.time())}.onnx",
        )
        input_tensor = (
            None  # TODO: Add your input example (ex: torch.randn(4, 2, 256, 256)).
        )
        torch.onnx.export(self.model, input_tensor, save_path)
        self.model_information[save_path] = metrics
        print(f"Model saved at {save_path}.")

    def export_model_mlflow(self, model_path: str) -> None:
        """Once the training session is finished this method will be called
        to export the best .onnx model in the mlflow running in the SharingHub.

        Args:
            model_path (str): The path to the .onnx file
        """
        onnx_model = onnx.load(model_path)
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="model",
            input_example=None,  # TODO: Add your input example (ex: torch.randn(4, 2, 256, 256)).
        )
        print("model saved in MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda, cpu, etc...)"
    )
    # TODO: Add your arguments.

    args = parser.parse_args()

    mlflow.set_tracking_uri("{{cookiecutter.mlflow_tracking_uri}}")
    mlflow.set_experiment(
        "example ({{cookiecutter.gitlab_project_id}})"
    )  # TODO: set your experiment.

    with mlflow.start_run():
        training_session = TrainingSession(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=args.save_dir,
            device=args.device,
        )
        training_session.train_and_valid()
