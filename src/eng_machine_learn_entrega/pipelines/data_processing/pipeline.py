from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_kobe_dataset,
    kobe_avarage_shot_artefact,
    build_kobe_model_pycaret
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_kobe_dataset,
                inputs="kobe@csv",
                outputs="kobe_shot:pandas.DataFrame",
                name="load_kobe_dataset",
            ),
            node(
                func=kobe_avarage_shot_artefact,
                inputs="kobe_shot",
                outputs="shots_made_and_missed",
                name="kobe_avarage_shot_artefact",
            ),
            node(
                func=build_kobe_model_pycaret,
                inputs="kobe_shot",
                outputs="kobe_shot_model",
                name="build_kobe_model_pycaret",
            ),
        ]
    )
