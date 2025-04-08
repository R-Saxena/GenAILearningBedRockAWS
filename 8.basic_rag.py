from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3

my_data = [
    "The weather is nice today",
    "Last night's game ended in a tie",
    "Rishabh's favourite IPL team is RCB",
    "Rishabh likes to eat pizza",
    "Rishabh likes to eat pasta"
]

question = "What are rishabh's likes ?"

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# New method replacing deprecated `get_relevant_documents`
retrieved_docs = retriever.invoke(question)

result_string = [doc.page_content for doc in retrieved_docs]

template = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's question based on the following context: {context}"),
        ("user", "{input}")
    ]
)

chain = template.pipe(model)

response = chain.invoke({
    "input": question,
    "context": result_string
})

print(response)
