from kfp.v2.dsl import component
from typing import NamedTuple


@component(packages_to_install=["google-cloud-aiplatform", "appengine-python-standard", "requests-toolbelt==0.10.1"],
           base_image="python:3.10")
def deploy_model_component(endpoint_id: str, project_id: str, location: str):
    """
    :param endpoint_id: str(int)
    :param project_id: str - example: projects/123/locations/us-central1/models/456
    :param location: str
    :return: remove deployed models from endpoint and deploy the latest trained model
    """
    import google.cloud.aiplatform as aip

    # connect to project
    aip.init(project=project_id, location=location)

    # get id of latest uploaded model
    list_o_models = aip.Model.list(order_by="create_time")
    latest_model = list_o_models[-1].resource_name

    # get deployed model ids from endpoint and undeploy them
    endpoint = aip.Endpoint(endpoint_id)
    deployed_model = endpoint.list_models()
    for dep_m in deployed_model:
        endpoint.undeploy(deployed_model_id=dep_m.id)

    # deploy latest uploaded model
    model = aip.Model(model_name=latest_model)
    model.deploy(endpoint=endpoint,
                 deployed_model_display_name="skl-activation",
                 traffic_split={"0": 100},
                 machine_type="n1-standard-4",
                 min_replica_count=1,
                 max_replica_count=1)


# eval func needs to contain everything it needs to run, as its made into a separate docker container
@component(packages_to_install=["pandas", "scikit-learn==1.0", "gcsfs", "joblib", "sqlalchemy", "tensorflow",
                                "appengine-python-standard", "requests-toolbelt==0.10.1",
                                "cloud-sql-python-connector", "pg8000", "google-cloud-firestore", "tensorflow-addons"],
           base_image="python:3.9")
def eval_component(framework: str) -> NamedTuple("outputs", [("deploy", str)]):
    """
    :return: str(bool) - pulls the latest trained model, evaluate it against a benchmark dataset, write down the result
    in an SQL db on GC, return "true" if the currently evaluated model is the current best
    """
    import gcsfs
    import joblib
    import sklearn
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    from google.cloud import firestore
    from google.cloud import storage
    from pathlib import Path

    FOLDER_SKL = 'sklearn-training'
    FOLDER_EVAL = "evaluation_data/"
    BUCKET_NAME = 'training-ml'

    SKL_METRICS = ["score", "specificity", "precision", "recall", "f1_score"]

    # firestore vars
    PROJECT = "disco-serenity-201413"
    COLLECTION = "ml"
    COUNTER = "counter"
    DATA_VERSION = "data_version"

    def get_latest_model_folder(framework):
        """
        :param framework: str: input framework - sklearn / eval
        :return: str: the name of the latest created folder
        """
        model_folders = []
        storage_client = storage.Client(PROJECT)
        bucket = storage_client.get_bucket(BUCKET_NAME)
        if framework == "sklearn":
            folder = FOLDER_SKL
        elif framework == "eval":
            folder = FOLDER_EVAL
        blobs = bucket.list_blobs(prefix=folder)

        for blob in blobs:
            folder_name = blob.name.split("/")[1]
            if "trainer" not in folder_name:  # ignore trainer files present in skl folder
                model_folders.append(folder_name)

        model_folders = sorted(model_folders)
        latest_folder = model_folders[-1]
        return latest_folder

    SKL_MODEL_URL = f"gs://training-ml/sklearn-training/{get_latest_model_folder('sklearn')}"
    EVAL_DATASET_URL = f"gs://training-ml/evaluation_data/{get_latest_model_folder('eval')}/eval_dataset.csv"

    class EvalMetrics:

        def __init__(self, project, collection, framework, eval_ds_url):
            self.data_version = None
            self.current_counter = None
            self.whole_data_dict = {}
            self.project = project
            self.framework = framework
            self.collection = collection
            self.db = firestore.Client(project=project)
            self.latest_model_folder = eval_ds_url[-2]

        def add_metrics(self, dict_o_metrics):
            """
            :param dict_o_metrics: dict
            :return: uploads the metrics to Firestore
            """
            self.get_info()
            doc_ref = self.db.collection(self.collection).document(str(self.current_counter + 1))
            dict_o_metrics.update({"data_version": self.data_version, "framework": self.framework,
                                   "model_folder": self.latest_model_folder})
            doc_ref.set(dict_o_metrics)
            self.update_counter()

        def pull_metrics(self):
            """
            :return: fetches metrics from Firestore
            """
            if self.current_counter is None:
                self.get_info()
            users_ref = self.db.collection(self.collection)
            # access the generator
            for doc in users_ref.stream():
                if doc.id != COUNTER and doc.id != DATA_VERSION:
                    self.whole_data_dict.update({doc.id: doc.to_dict()})

        def get_info(self):
            """
            :return: fetches the counter (so it can be updated later) and latest dataset version
            """
            doc_ref = self.db.collection(self.collection).document(COUNTER)
            doc = doc_ref.get()
            self.current_counter = doc.to_dict()[COUNTER]

            doc_ref = self.db.collection(self.collection).document(DATA_VERSION)
            doc = doc_ref.get()
            self.data_version = doc.to_dict()[DATA_VERSION]

        def update_counter(self):
            count_val = {COUNTER: self.current_counter + 1}
            doc_ref = self.db.collection(self.collection).document(COUNTER)
            doc_ref.set(count_val)

        def compare_metrics(self):
            """
            :return: compares two sets of metrics and returns the best acc
            """
            if self.current_counter is None:
                self.pull_metrics()

            df = pd.DataFrame.from_dict(self.whole_data_dict, orient="index")
            # get data from the latest data version
            current_data = df[df[DATA_VERSION] == self.data_version]
            # highest accuracy, convert to dict
            best_acc = current_data.loc[current_data["score"].idxmax()].to_dict()
            return best_acc

    def load_joblib(bucket_name, file_name):
        """
        :param bucket_name: bucket where model is
        :param file_name: name of model
        :return: loads an sklearn model saved as joblib file from google cloud storage bucket
        """
        fs = gcsfs.GCSFileSystem()
        with fs.open(f'{bucket_name}/{file_name}') as f:
            return joblib.load(f)

    def skl_evaluate(model, validation_data, validation_labels, metrics):
        """
        :param model: skl model file
        :param validation_data: df
        :param validation_labels: df
        :param metrics: list["metric", "names"]
        :return:
        """
        predicted_labels = model.predict(validation_data)
        result_dict = {}
        if "score" in metrics:
            acc = model.score(validation_data, validation_labels)
            result_dict.update({"score": acc})
        if "specificity" in metrics:
            tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_labels).ravel()
            specificity = tn / (tn + fp)
            result_dict.update({"specificity": specificity})
        if "precision" in metrics:
            precision = precision_score(validation_labels, predicted_labels, average="macro")
            result_dict.update({"precision": precision})
        if "recall" in metrics:
            recall = recall_score(validation_labels, predicted_labels, average="macro")
            result_dict.update({"recall": recall})
        if "f1" in metrics:
            f1_scor = f1_score(validation_labels, predicted_labels, average="macro")
            result_dict.update({"f1_score": f1_scor})

        return result_dict

    # load eval dataset
    dataset = pd.read_csv(EVAL_DATASET_URL)
    data_label = dataset["Attrition"]
    data_eval = dataset.drop(columns="Attrition")

    # load model
    if framework == "sklearn":
        skl_model = load_joblib(SKL_MODEL_URL, "model.joblib")
        eval_result_dict = skl_evaluate(skl_model, data_eval, data_label, SKL_METRICS)
    else:
        assert 0, "Unknown framework"

    em = EvalMetrics(PROJECT, COLLECTION, framework, EVAL_DATASET_URL)
    em.pull_metrics()
    em.add_metrics(eval_result_dict)
    best_model = em.compare_metrics()

    if best_model["framework"] == framework:
        return ("true",)
    else:
        return ("false",)
