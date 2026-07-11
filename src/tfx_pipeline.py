"""
tfx_pipeline.py — TFX pipeline for the ranking model + Kubeflow compile.

    python src/tfx_pipeline.py                # run locally (LocalDagRunner)
    python src/tfx_pipeline.py --compile-kfp  # emit ott_pipeline.yaml (Kubeflow Pipelines v2 IR)

Components: ExampleGen -> StatisticsGen -> SchemaGen -> ExampleValidator
            -> Transform -> Trainer -> Evaluator -> Pusher

The Trainer module (trainer_module.py, sibling file) builds a small Keras ranking model.
This is the "data pipeline engineering" + "experimentation (Evaluator)" story from the JD,
and the SAME graph compiles to Kubeflow with a one-line runner swap.
"""
import os
import sys

from tfx import v1 as tfx
from tfx.proto import example_gen_pb2

PIPELINE_NAME = "ott_ranking"
DATA_ROOT = os.path.abspath("data/processed")          # holds interactions.tfrecord
MODULE = os.path.abspath("src/trainer_module.py")
PIPELINE_ROOT = os.path.abspath(f"tfx/{PIPELINE_NAME}")
METADATA = os.path.join(PIPELINE_ROOT, "metadata.db")
SERVING = os.path.abspath("artifacts/ranking_tf")


def build_pipeline(pipeline_root, metadata_connection=None) -> tfx.dsl.Pipeline:
    example_gen = tfx.components.ImportExampleGen(
        input_base=DATA_ROOT,
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name="single", pattern="interactions.tfrecord"),
        ]),
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ])))

    stats = tfx.components.StatisticsGen(examples=example_gen.outputs["examples"])
    schema = tfx.components.SchemaGen(statistics=stats.outputs["statistics"],
                                      infer_feature_shape=True)
    validator = tfx.components.ExampleValidator(statistics=stats.outputs["statistics"],
                                                schema=schema.outputs["schema"])
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema.outputs["schema"],
        module_file=MODULE)
    trainer = tfx.components.Trainer(
        module_file=MODULE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema.outputs["schema"],
        train_args=tfx.proto.TrainArgs(num_steps=2000),
        eval_args=tfx.proto.EvalArgs(num_steps=500))
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"])
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(base_directory=SERVING)))

    components = [example_gen, stats, schema, validator, transform,
                  trainer, evaluator, pusher]
    return tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection)


def run_local():
    md = tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA)
    p = build_pipeline(PIPELINE_ROOT, md)
    tfx.orchestration.LocalDagRunner().run(p)


def compile_kfp(out="ott_pipeline.yaml"):
    """Compile the identical DAG to Kubeflow Pipelines v2 IR (runs on Vertex AI / GKE KFP)."""
    from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as kfp_runner
    runner = kfp_runner.KubeflowV2DagRunner(
        config=kfp_runner.KubeflowV2DagRunnerConfig(
            default_image="gcr.io/tfx-oss-public/tfx:1.15.1"),
        output_filename=out)
    runner.run(build_pipeline(pipeline_root=f"gs://YOUR_BUCKET/{PIPELINE_NAME}"))
    print(f"Compiled Kubeflow v2 pipeline → {out}")


if __name__ == "__main__":
    if "--compile-kfp" in sys.argv:
        compile_kfp()
    else:
        run_local()
