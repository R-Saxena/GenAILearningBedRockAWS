import boto3
import json

client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

def get_config(prompt:str):
    
    return json.dumps({
        "inputText": prompt,
        "textGenerationConfig":{
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1
        }
    })

print("Bot: Hello! I am your assistant tell me what do u want to talk about.")


while True:
    user_input = input('User: ')
    if user_input.lower() == "exit":
        break
    response = client.invoke_model(
        body = get_config(user_input),
        modelId = "amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    print(response_body.get('results')[0].get('outputText'))

#tell me about rishabh saxena
# tell me more 