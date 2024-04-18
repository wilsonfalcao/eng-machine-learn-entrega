from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_kobe_dataset,
    train_kobe_dataset,
    kobe_avarage_shot_artefact,
    build_kobe_model_pycaret,
    kobe_histogram
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_kobe_dataset,
                inputs="kobe@csv",
                outputs="dataset_kobe_prod",
                name="PreparacaoDados",
            ),
            node(
                func=kobe_avarage_shot_artefact,
                inputs="dataset_kobe_prod",
                outputs="shots_made_and_missed",
                name="kobe_avarage_shot_artefact",
            ),
            node(
                func=train_kobe_dataset,
                inputs="dataset_kobe_prod",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_kobe_dataset",
            ),
            node(
                func=build_kobe_model_pycaret,
                inputs="dataset_kobe_prod",
                outputs="kobe_shot_model",
                name="build_kobe_model_pycaret",
            ),
        ]
    )
