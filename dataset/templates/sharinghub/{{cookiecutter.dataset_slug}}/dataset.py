import datasets
from dvc.api import DVCFileSystem


class CustomBuilderConfig(datasets.BuilderConfig):
    """
    This class is used to transfer configurations variables (such as 'no_cache'
    option and 'context' path) to the class Sen1floods11Dataset in self config
    field.
    """

    def __init__(self, version="1.0.0", description=None, **kwargs):
        super().__init__(version=version, description=description)
        config = kwargs.get("config_kwargs")
        self.no_cache = config["no_cache"]
        self.context = config["context"]


class MyCustomDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fs = DVCFileSystem(
            self.config.context
        )  # Note: It is possible to dl a file with self.fs.read_bytes(path_to_file)

    def _info(self):
        """Contains informations and typings for the dataset."""
        return datasets.DatasetInfo(
            description="A custom dataset",  # TODO: add description
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(
                        names=["negative", "neutral", "positive"]
                    ),
                }
            ),  # TODO: Define your own features of your custom dataset.
        )

    def _split_generators(self, dl_manager):
        """Downloads the data and defines train/test splits."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={}
            ),  # Add the path to the specified split (can be a .csv, .parquet, etc ...)
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={}
            ),  # The path to the specified split will be stored to a argument that will be passed to
            # _generate_examples method to load the split. EXAMPLE: {"file_path": "path_to_file.csv"}
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={}),
            # Note: "file_path" is the argument name. You can use a different one if you want.
            # Just make sure that _generate_examples method get back the right argument.
        ]

    def _generate_examples(self, **kwargs) -> I:
        """Reads the dataset and yields examples.
        This method is called each time we are iterating in the dataset,
        returns the downloaded images.
        """
        # TODO: complete the _generate_examples method.
        file_path = kwargs.get("file_path")
        with open(file_path, "r", encoding="utf-8") as f:
            pass
