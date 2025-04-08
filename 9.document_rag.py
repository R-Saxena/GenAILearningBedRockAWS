from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name = "bedrock-runtime", region_name = AWS_REGION)

model = BedrockLLM(model_id = "amazon.titan-text-express-v1", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

question = "Astrology is really effective ?"

#data ingestion
loader = PyPDFLoader("pdf_data/astrology.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(separators = ["\n"])
splitted_docs = splitter.split_documents(documents)

vector_store = FAISS.from_texts(splitted_docs, bedrock_embeddings)

splitter = RecursiveCharacterTextSplitter(separators=["\n"])
splitted_docs = splitter.split_documents(documents)

# Extract pure text from documents (no Document objects here!)
texts = [doc.page_content for doc in splitted_docs]

print(f"Sample type: {type(texts[0])}")
print(f"Sample content: {texts[0][:200]}")

try:
    vector_store = FAISS.from_texts(texts, bedrock_embeddings)
except Exception as e:
    print(f"Embedding failed: {e}")
    print(f"First item type: {type(texts[0])}")
    print(f"First item content: {texts[0]}")
    raise


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