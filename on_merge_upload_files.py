from google.cloud import storage
from google.cloud import aiplatform
import shutil
import os

# google storage
GCS_SKL_PATH = "gs://training-ml/sklearn-training/trainer-0.1.0.tar.gz"
GCS_PIPELINES_PATH = "gs://training-ml/main.zip"

# local file paths
SKL_LOCAL_PATH = "sklearn-training/dist/trainer-0.1.0.tar.gz"
MAKE_ZIP_FROM = "cloudFuncPipelines/"

# names
UPLOAD_ZIP_FROM = "main.zip"
ZIP_NAME = "main"

# zip the sk-learn trainer and upload to google cloud storage
os.system("cd sklearn-training && python setup.py sdist --formats=gztar")
blob = storage.blob.Blob.from_string(GCS_SKL_PATH, client=storage.Client())
blob.upload_from_filename(SKL_LOCAL_PATH)

# zip the pipelines files
shutil.make_archive(ZIP_NAME, 'zip', MAKE_ZIP_FROM)
blob = storage.blob.Blob.from_string(GCS_PIPELINES_PATH, client=storage.Client())
blob.upload_from_filename(UPLOAD_ZIP_FROM)

# create service account credentials
aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location='europe-west3')

# create cloud function (so it uses the new pipelines/trainer files)
cloud_function = "gcloud functions deploy pipeline-trigger --project disco-serenity-201413 --region europe-west3 " \
                 "--entry-point pipeline --runtime python310 --trigger-event google.storage.object.finalize " \
                 "--trigger-resource gs://data_bucket42 --source gs://training-ml/main.zip --service-account " \
                 "simeon@disco-serenity-201413.iam.gserviceaccount.com"
# run gcloud command
os.system(cloud_function)
