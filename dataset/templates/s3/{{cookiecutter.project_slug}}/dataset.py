"""Module compatible with `load_dataset` method from `datasets` library."""

import datasets
from dvc.api import DVCFileSystem

_DESCRIPTION = "{{ cookiecutter.dataset_description }}"


class DatasetBuilderConfig(datasets.BuilderConfig):
    """
    This class is used to transfer configurations variables
    (such as 'no_cache' option and 'context' path) in Dataset
    self.config field.
    """

    def __init__(self, version="1.0.0", description=None, **kwargs):
        super().__init__(version=version, description=description)
        config = kwargs.get("config_kwargs")
        self.no_cache = config["no_cache"]
        self.context = config["context"]


class Dataset(datasets.GeneratorBasedBuilder):
    """Custom dataset class for loading/preprocessing the {{ cookiecutter.dataset_name }} dataset.

    This class uses the Hugging Face `datasets` library to handle data
    generation and implements specific logic to load, process, and stream the
    dataset efficiently.
    """

    BUILDER_CONFIG_CLASS = DatasetBuilderConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Note: It is possible to download a file from DVC with
        # self.fs.read_bytes(repo_relative_path_to_file)
        self.fs = DVCFileSystem(self.config.context)

    def _info(self):
        """Contains informations and typings for the dataset."""
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(  # TODO: Define your own features for your dataset.
                {
                    "id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(
                        names=["negative", "neutral", "positive"]
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        """Downloads the data and defines train/test splits."""
        # TODO: Add the path for each the specified split (can be a .csv, .parquet, etc ...)
        # The path to the specified split will be stored in an argument that will be passed to
        # _generate_examples method to load the split. EXAMPLE: {"file_path": "path_to_file.csv"}
        # Note: "file_path" is the argument name. You can use a different one if you want.
        # Just make sure that _generate_examples method get back the right argument.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={},
            ),
        ]

    def _generate_examples(self, **kwargs):
        """Reads the dataset and yields examples.

        This method is called each time we are iterating in the dataset,
        returns the downloaded images.
        """
        # TODO: complete the _generate_examples method.
        file_path = kwargs.get("file_path")
        with open(file_path, "r", encoding="utf-8") as f:
            pass
