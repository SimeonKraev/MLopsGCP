# Essentially a copy of the pipelines file -> to be triggered by Google Cloud Functions
import google.cloud.aiplatform as aip
import tempfile
import datetime
import kfp
import os
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google_cloud_pipeline_components.types import artifact_types
from kfp.v2.components import importer_node
from kfp.v2 import compiler, dsl
from google.cloud import storage
from custom_components import eval_component, deploy_model_component

# Variables
LOCATION = "europe-west3"
PROJECT_ID = "disco-serenity-201413"
BUCKET = "gs://training-ml/"
PIPELINE_BUCKET = "gs://pipeline-jobs/"
BUCKET_NAME = 'training-ml'
FOLDER_SKL = 'sklearn-training'
ENDPOINT = "6115685983829622784"

framework = "sklearn"  # or sklearn
latest_model = f"model_training_{str(datetime.date.today())}"


def bucket_path(path):
    b_path = os.path.join(BUCKET, path)
    return b_path


sklearn_params = {
    "display_name": "sklearn-activation",
    "pipeline_root_path": PIPELINE_BUCKET,
    "train_dir": bucket_path("sklearn-training"),
    "model_path": bucket_path(f"sklearn-training/{latest_model}/"),
    "staging_bucket": bucket_path("sklearn-training/staging"),
    "python_package_gs_uri": bucket_path("sklearn-training/trainer-0.1.0.tar.gz"),
    "training_container_uri": "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest",
    # tf contains sklearn, it's preferred due to gcsfs
    "prediction_container_uri": "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    "python_module_name": "trainer.task",
}


def get_latest_model_folder():
    """
    :return: str: the name of the latest created folder
    """
    model_folders = []
    storage_client = storage.Client(PROJECT_ID)
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=FOLDER_SKL)

    for blob in blobs:
        if "model_training" in blob.name:
            folder_name = blob.name.split("/")[1]
            model_folders.append(folder_name)

    model_folders = sorted(model_folders)
    latest_folder = model_folders[-1]

    return latest_folder


class Config:
    def __init__(self, project_id, location, endpoint, name, ml_framework):
        self.project_id = project_id
        self.location = location
        self.name = name
        self.framework = ml_framework
        self.endpoint = endpoint
        init_params = self.get_init_params()
        self.setup(**init_params)

    def get_init_params(self):
        if self.framework == "sklearn":
            return sklearn_params
        else:
            raise ValueError("framework not supported")

    def setup(self,
              display_name,
              model_path,
              training_container_uri,
              prediction_container_uri,
              pipeline_root_path,
              train_dir,
              staging_bucket,
              python_package_gs_uri,
              python_module_name,
              **kwargs
              ):
        self.display_name = display_name
        self.model_path = model_path
        self.training_container_uri = training_container_uri
        self.prediction_container_uri = prediction_container_uri
        self.pipeline_root_path = pipeline_root_path
        self.train_dir = train_dir
        self.staging_bucket = staging_bucket
        self.python_package_gcs_uri = python_package_gs_uri
        self.python_module_name = python_module_name


cfg = Config(PROJECT_ID, LOCATION, ENDPOINT, f"training-{framework}", framework)


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(name=cfg.name, pipeline_root=cfg.pipeline_root_path)
def pipeline(project_id: str, framework: str, location: str = LOCATION):
    """
    Trains a model
    Evaluates it
    Deploys the uploaded model to the new endpoint
    """

    # Model Training
    training_job_run_op = gcc_aip.CustomPythonPackageTrainingJobRunOp(
        project=cfg.project_id,
        location=cfg.location,
        display_name=cfg.display_name,
        python_package_gcs_uri=cfg.python_package_gcs_uri,
        python_module_name=cfg.python_module_name,
        container_uri=cfg.training_container_uri,
        base_output_dir=cfg.train_dir,
        staging_bucket=cfg.staging_bucket,
        model_labels=cfg.framework,
        machine_type="n1-standard-4",
        replica_count=1,
    )

    # To upload a custom trained model we need to pair it with pre-build container
    unmanaged_model_importer = importer_node.importer(
        artifact_uri=f"{BUCKET}{FOLDER_SKL}/{get_latest_model_folder()}",
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": cfg.prediction_container_uri
            }
        },
    )

    # Upload the trained model
    model_upload_op = gcc_aip.ModelUploadOp(
        project=cfg.project_id,
        location=cfg.location,
        display_name=cfg.display_name,
        unmanaged_container_model=unmanaged_model_importer.outputs["artifact"],
    )
    model_upload_op.after(training_job_run_op)

    # evaluation here, trigger the below OPs iff cond is true
    eval_result = eval_component(framework)
    eval_result.after(model_upload_op)

    with dsl.Condition(eval_result.outputs["deploy"] == "true"):
        deploy_model_component(cfg.endpoint, cfg.project_id, cfg.location)


tmpdir = tempfile.gettempdir()  # a temp dir to store the compiled pipeline
print("starting to compile...")
compiler.Compiler().compile(pipeline_func=pipeline, package_path=tmpdir+'/pipelines_v1.json')
print("finished compiling!")

job = aip.PipelineJob(
    display_name="skl-training-pipeline-v1",
    pipeline_root=cfg.pipeline_root_path,
    template_path=tmpdir+"/pipelines_v1.json",
    location=cfg.location,
    parameter_values={'project_id': cfg.project_id, 'framework': framework}
)

job.submit()
print("The job is submitted!")
