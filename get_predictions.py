from google.cloud import aiplatform
from google.cloud import storage


def endpoint_predict_sample(project: str, location: str, instances: list, endpoint: str):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction


def test_gcp_access():
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)


instance = [{"key": [-1.4151810691502822, 2.0, -1.3944595889025913, 0.0, 1.5803695257836163, -1.8684257522543597, 0.0,
                     0.0, 1.1450852950444137, 1.1697805251007627, 1.0, -0.3882963188761366, -2.432005601372186,
                     -0.9614863916702531, 1.0, 0.24620020465769926, 1.0, -1.0513297699033917, -0.38341020430999717,
                     -0.6780493930322936, 0.0, 0.0, -1.1505541015749308, -0.42623001504290287, 0.26623257679518353, 0.0,
                     0.24198831185855807, -1.3216011591512031, -0.6201892226844657, 0.33809616377248186,
                     -0.9810141559830953, -1.1676872598353414, -0.6791456840737199, -1.1559347102263289
                     ]}]

# pred = endpoint_predict_sample("disco-serenity-201413", "europe-west3", instance, "7538849854357766144")
# print(pred)