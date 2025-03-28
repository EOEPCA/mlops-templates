import argparse
import time
import uuid
from pathlib import Path

import mlflow
import onnx
import tensorflow as tf
import tf2onnx
from datasets import load_dataset


class TrainingSession:
    def __init__(
        self, epochs=10, batch_size=32, lr=0.001, save_dir="checkpoints", device="auto"
    ):
        self.model_information = {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir

        # Set device based on user input
        if device.lower() == "gpu" and tf.config.list_physical_devices("GPU"):
            self.device = "GPU"
            for d in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(d, True)
        else:
            self.device = "CPU"
        print(f"Using device: {self.device}")

        # Load dataset using Hugging Face datasets
        train_dataset = load_dataset(
            "dataset.py",  # TODO: Add your dataset.py path here.
            split="train",
            streaming=True,
            trust_remote_code=True,
            config_kwargs={},  # TODO: Add your dataset config here.
        )
        train_dataset = train_dataset.with_format("numpy")
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_dataset,
            output_signature=(
                tf.TensorSpec(
                    shape=(None, 28, 28), dtype=tf.float32
                ),  # TODO: Adjust shape
                tf.TensorSpec(
                    shape=(None,), dtype=tf.int64
                ),  # Adjust label type if needed
            ),
        )
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        self.train_dataset = train_dataset

        # Define model
        self.model = None  # TODO: Implement build_model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss="sparse_categorical_crossentropy",  # TODO: Set your loss function
            metrics=["accuracy"],  # TODO: Define evaluation metrics
        )

    def train_and_valid(self):
        self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
        )
        # TODO: Add validation step if necessary

        # Save model after training
        self.save_model()

    def save_model(self):
        save_path = Path(
            self.save_dir, f"model_{str(uuid.uuid4())[:8]}_{int(time.time())}"
        )
        spec = None  # TODO: Add your input example.
        onnx_model, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=spec, opset=13
        )
        onnx.save(onnx_model, str(save_path))

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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device to use: 'cpu', 'gpu', or 'auto'",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri("{{cookiecutter.mlflow_tracking_uri}}")
    mlflow.set_experiment("example ({{cookiecutter.gitlab_project_id}})")

    with mlflow.start_run():
        training_session = TrainingSession(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=args.save_dir,
            device=args.device,
        )
        training_session.train_and_valid()
