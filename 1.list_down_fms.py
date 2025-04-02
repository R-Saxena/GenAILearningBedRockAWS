import boto3
import pprint

bedrock = boto3.client(
    service_name = 'bedrock',
    region_name = 'us-east-1'
)


pp = pprint.PrettyPrinter(depth=4)

def list_foundation_model():
    models = bedrock.list_foundation_models()
    for model in models["modelSummaries"]:
        pp.pprint(model)
        pp.pprint("-------------------------")

def get_foundation_model(modelIdentifier):

    model = bedrock.get_foundation_model(modelIdentifier = modelIdentifier)
    pp.pprint(model)

get_foundation_model('mistral.mistral-7b-instruct-v0:2')
