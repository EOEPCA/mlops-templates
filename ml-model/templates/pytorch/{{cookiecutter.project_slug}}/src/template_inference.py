import mlflow


class MLflowONNXInference:
    def __init__(self, tracking_uri: str, model_name: str, model_version: str = None):
        """
        Initialize MLflow ONNX inference pipeline.
        :param tracking_uri: URI of the MLflow tracking server.
        :param model_name: Registered name of the model in MLflow.
        :param model_version: Specific version of the model to load (optional).
        """
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_version = model_version
        self.model = None

    def load_model(self) -> None:
        """
        Load the ONNX model from the MLflow tracking server.
        """
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            if self.model_version:
                model_uri = f"models:/{self.model_name}/{self.model_version}"
            else:
                model_uri = f"models:/{self.model_name}/latest"

            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model loaded successfully from {model_uri}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")

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
    # Configuration
    tracking_uri = "{{cookiecutter.mlflow_tracking_uri}}"
    model_name = (
        "your-model-name"  # TODO: Replace with your registered model name in MLflow
    )
    model_version = "1"  # Optional: Specify model version, or None for latest version

    inference_pipeline = MLflowONNXInference(tracking_uri, model_name, model_version)

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
