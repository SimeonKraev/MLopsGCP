# aurora-ml
ML infrastructure on GCP for Crayon, using Vertex AI on GCP



Flow:
1. Pushing to git triggers GitActions to run some tests and checks (not implemented)
2. When merged in master would trigger a new pipeline build (not implemented)
3. On uploading new dataset to GCS, it triggers Cloud Functions which start the pipeline (implemented)
4. Pipeline flow: (implemented)
   1. Train model
   2. Upload model to Vertex model registry
   3. Evaluate the model
         1. Evaluate the model on an eval dataset
         2. Write down metrics for the current training to Firestore
         3. Pull the metrics of the current best model and compare
         4. If new model's performance is better -> deploy to endpoint OR follow next step
   4. (Optional) Split traffic between the currently deployed model and the newly trained
      1. Gather data about their performance and evaluate next steps


Project structure:

CloudFuncPipelines - files required to set up a Cloud Function on GCP, which triggers when a file is uploaded in the
bucket, which then triggers the pipeline

sklearn-training - make a package for training custom containers on Vertex AI
 