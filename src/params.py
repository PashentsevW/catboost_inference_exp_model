from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, YamlConfigSettingsSource


class DatasetParams(BaseModel):
    n_samples: int
    n_features: int
    n_classes: int
    random_state: int


class CatboostModelParams(BaseModel):
    loss_function: str
    iterations: int
    learning_rate: float
    random_state: int
    verbose: int


class PipelineParams(BaseSettings):
    random_state: int
    dataset_params: DatasetParams
    test_size: float
    catboost_model_params: CatboostModelParams
    eval_metrics: list[str]

    @classmethod
    def load_from_yaml(cls, yaml_file: Path | str) -> "PipelineParams":
        source = YamlConfigSettingsSource(cls, yaml_file=yaml_file)
        return PipelineParams(**source())
