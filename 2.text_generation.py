import boto3
import json
import pprint

bedrock = boto3.client(
    service_name = 'bedrock-runtime',
    region_name = 'us-east-1'
)

model_id = "amazon.titan-text-express-v1"

model_config = json.dumps({
    "inputText": "Tell me about GenAI",
    "textGenerationConfig": {
        "maxTokenCount": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1
    }
})


response = bedrock.invoke_model(
    body = model_config,
    modelId = model_id,
    accept = "application/json",
    contentType = "application/json"
)

response_body = json.loads(response.get('body').read())

pp = pprint.PrettyPrinter(depth=4)

pp.pprint(response_body.get('results'))