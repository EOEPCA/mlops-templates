import argparse
import sys

import mlflow
import onnxruntime as ort


class MLflowONNXInference:
    def __init__(
        self,
        model_path: str | None = None,
        model_uri: str | None = None,
        tracking_uri: str | None = None,
    ):
        """
        Initialize MLflow ONNX inference pipeline.
        :param tracking_uri: URI of the MLflow tracking server.
        :param model_name: Registered name of the model in MLflow.
        :param model_version: Specific version of the model to load (optional).
        """
        self.tracking_uri = tracking_uri
        self.model_uri = model_uri
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """
        Load the ONNX model from the MLflow tracking server.
        """
        if self.model_path:
            self.model = ort.InferenceSession(self.model_path)
        elif self.model_uri:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                self.model = mlflow.pyfunc.load_model(self.model_uri)
                print(f"Model loaded successfully from {self.model_uri}")
            except Exception as e:
                raise RuntimeError(f"Failed to load the model: {e}")
        else:
            raise ValueError(f"{self.model_path} and {self.model_uri} are both None.")

    def preprocess(self, raw_data):
        """
        Preprocess raw input data to match the model's input requirements.
        # TODO: Customize this method to handle your specific input data format.
        :param raw_data: Raw input data (e.g., image, text, etc.).
        :return: Preprocessed input as a numpy array or pandas DataFrame.
        """
        raise NotImplementedError("Preprocessing function needs to be implemented.")

    def postprocess(self, model_output):
        """
        Postprocess model outputs to generate human-readable results.
        # TODO: Customize this method based on your output format.
        :param model_output: Raw model output.
        :return: Postprocessed results.
        """
        raise NotImplementedError("Postprocessing function needs to be implemented.")

    def infer(self, preprocessed_data):
        """
        Run inference on the preprocessed data.
        :param preprocessed_data: Preprocessed input data.
        :return: Raw output from the model.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call `load_model` first.")

        try:
            return self.model.predict(preprocessed_data)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    def run(self, raw_data):
        """
        High-level method to execute the full pipeline: preprocess → infer → postprocess.
        :param raw_data: Raw input data (e.g., image, text).
        :return: Postprocessed results.
        """
        preprocessed_data = self.preprocess(raw_data)
        raw_output = self.infer(preprocessed_data)
        results = self.postprocess(raw_output)
        return results


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--model-path",
        help="Path to the model if it's stored locally.",
    )
    parser.add_argument("-u", "--model-uri", help="Mlflow model URI.")
    args = parser.parse_args()

    # Configuration
    model_path = args.model_path
    model_uri = args.model_uri
    tracking_uri = "{{cookiecutter.mlflow_tracking_uri}}"

    if model_path:  # If the model is already stored locally, usefull for Docker usage.
        inference_pipeline = MLflowONNXInference(model_path=model_path)
    elif model_uri:
        inference_pipeline = MLflowONNXInference(
            model_uri=model_uri, tracking_uri=tracking_uri
        )
    else:
        print("ERROR: one of the options --model-path or --model-uri are required.")
        sys.exit(-1)

    # Load the model
    inference_pipeline.load_model()

    # Example input (to be replaced by actual data)
    raw_data = None  # TODO: Provide your raw input data here

    try:
        results = inference_pipeline.run(raw_data)
        print("Inference results:", results)
    except NotImplementedError as e:
        print(f"Error: {e} - You need to implement the missing methods.")
    except Exception as e:
        print(f"Unexpected error: {e}")
