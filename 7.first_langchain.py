from langchain_community.llms import Bedrock
import boto3
from langchain_core.prompts import ChatPromptTemplate


AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name = "bedrock-runtime", region_name = AWS_REGION)

model = Bedrock(model_id = "amazon.titan-text-express-v1", client = bedrock)

def invoke_model():
    response = model.invoke("what is the highest mountain in the world")
    print(response)
    return response

def first_chain():
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "write a short description for the product provided the user"),
            ("human", "{product_name}")
        ]
    )

    chain = template.pipe(model)

    response = chain.invoke({
        "product_name": "bicycle"
    })

    print(response)

    return response

invoke_model()
first_chain()
