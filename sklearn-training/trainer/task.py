import pandas as pd
import datetime
import joblib
import os

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage


PROJECT_ID = "disco-serenity-201413"
BUCKET = "gs://training-ml/"
BUCKET_NAME = 'training-ml'
DATA_URL = "gs://data_bucket42/train_preprocessed.csv"
NEW_FOLDER_NAME = f"model_training_{str(datetime.date.today())}"
NEW_FOLDER_PATH = f"sklearn-training/{NEW_FOLDER_NAME}/"
OUTPUT_MODEL = f"gs://training-ml/sklearn-training/{NEW_FOLDER_NAME}/"
MODEL_NAME = 'model.joblib'


class SklearnModel:
    """
    All the basic functions that you would need for your ML model
    """

    def __init__(self, model_name):
        self.train_labels = None
        self.train_data = None
        self.model_name = model_name
        self.model_prediction_prob = None
        self.validation_labels = None
        self.model_prediction = None
        self.validation_data = None
        self.test_labels = None
        self.parameters = None
        self.test_data = None
        self.proba = True

        if model_name == "random_forest":
            self.model = RandomForestClassifier(max_depth=3, random_state=0)
            self.parameters = {
                'n_estimators': (10, 20, 30, 40, 60, 80, 100, 120, 140),
                'criterion': ("gini", "entropy", "log_loss"),
                'max_features': ("sqrt", "log2", None)
            }

    def optimize(self, dataset):
        """
        :param dataset: data
        :return: runs a grid seach over the model hyperparameters
        """
        train_data, _, train_labels, _, _, _ = dataset.data_split()
        grd = GridSearchCV(self.model, self.parameters, cv=5)
        grd.fit(train_data, train_labels["activated"].values)
        self.model = self.model.set_params(
            **grd.best_params_)  # set best params of grid search
        print(f"Best params set to model: {grd.best_params_}")

    def train(self, dataset, label_column):
        y = dataset[label_column]
        X = dataset.drop(columns=label_column)
        self.train_data, self.validation_data, self.train_labels, self.validation_labels = train_test_split(
            X, y, test_size=0.10, shuffle=True)  # , random_state=42
        self.model.fit(self.train_data, self.train_labels.values)
        print("num features", len(self.train_data.columns))

    def score(self):
        """
        :return: prints out model accuracy
        """
        score = self.model.score(self.validation_data, self.validation_labels)
        print(f"{self.model_name}'s mean accuracy on validation data: {score}")

    def save_model(self, path="data/model.joblib"):
        joblib.dump(self.model, path)
        print(f"{self.model_name} model saved at {path}")

    def load_model(self, path=""):
        self.model = joblib.load(path)
        print(f"{self.model_name} model was loaded from {path}")

    def eval(self):
        """
        :return: prints out the cross validation score
        """
        score = cross_val_score(self.model,
                                self.test_data,
                                self.test_labels["activated"].values,
                                cv=5)
        print(f"Cross validation score {score}")

    def predict(self, data_to_predict):
        """
        :param data_to_predict: data
        :return: prints out prediction results
        """
        self.model_prediction = self.model.predict(data_to_predict)
        print(f"{self.model_name} model predicted: {self.model_prediction}")

    def predict_prob(self, data_to_predict):
        """
        :param data_to_predict: data
        :return: prints out prediction results
        """
        if self.proba:
            self.model_prediction_prob = self.model.predict_proba(data_to_predict)
            print(
                f"{self.model_name} model predicted probabilities {self.model_prediction_prob}"
            )
        else:
            print(
                f"{self.model_name} model does not support probabilistic predictions, use .predict instead"
            )

    def save_pred(self, prob=False, path="data/"):
        """
        :param prob: bool
        :param path: str
        :return: save prediction probabilities
        """
        path = path + self.model_name
        if prob:
            path = path + "_pred_prob.csv"
            pd.DataFrame(self.model_prediction_prob,
                         columns=["not_activated", "activated"]).to_csv(path)
        else:
            path = path + "_pred.csv"
            pd.DataFrame(self.model_prediction,
                         columns=["activated"]).to_csv(path)

    def feature(self):
        important = self.model.feature_importances_
        imp_features = pd.DataFrame(important,
                                    index=[self.train_data.columns],
                                    columns=['importance']).sort_values(by='importance')
        return imp_features


def create_gcs_folder(bucket_name, destination_folder_name):
    """
    :param bucket_name: str
    :param destination_folder_name: str
    :return: Creates a folder in Google Cloud Storage
    """
    storage_client = storage.Client(PROJECT_ID)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_folder_name)
    blob.upload_from_string('')
    print(f"Created folder {destination_folder_name}")


def get_data():
    """
    :return: read csv from GCS bucket
    """
    return pd.read_csv(DATA_URL)


def preprocess(df, preprocess_dict):
    """
    :param df: DataFrame
    :param preprocess_dict: dict{"transformer": ["columnNames", "toPreprocess"]}
    :return: preprocessed DataFrame
    """
    for col in df.columns:
        if col in preprocess_dict["ordinal"]:
            df[col] = OrdinalEncoder().fit_transform(df[[col]])
        elif col in preprocess_dict["numeric"]:
            df[col] = StandardScaler().fit_transform(df[[col]])
    return df


def build_model(df, label_column, model="random_forest"):
    """
    :param df: DataFrame
    :param label_column: DataFrame
    :param model: str
    :return: trains model and uploads it to GCS
    """
    print("Building the model...")
    clf = SklearnModel(model)
    print("Training started...")
    clf.train(df, label_column)

    # Save model artifact to local filesystem
    print("Saving model...")
    joblib.dump(clf.model, MODEL_NAME)
    clf.save_model(MODEL_NAME)

    # Upload model artifact to Cloud Storage
    create_gcs_folder(BUCKET_NAME, NEW_FOLDER_PATH)
    storage_path = os.path.join(OUTPUT_MODEL, MODEL_NAME)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    print("Uploading model to " + MODEL_NAME)
    blob.upload_from_filename(MODEL_NAME)
    print("Finished")


def main():
    df = get_data()
    build_model(df, "Attrition")


main()
