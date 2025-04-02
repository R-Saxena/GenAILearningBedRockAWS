from similiarity import cosineSimilarity
import boto3
import json
import base64

client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

images = [
    'images/animal_group.jpg',
    'images/cats.jpg',
    'images/lion.jpg',
    'images/tiger.png',
    'images/zebra.jpg',
]

def get_embedding(input_image_path: str):

    with open(input_image_path, "rb") as f:
        input_image = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = client.invoke_model(
            body=json.dumps({"inputText": input_image}),
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding", [])
    except Exception as e:
        print(f"Error fetching embedding for '{input_image}': {e}")
        return None
    

test_image = "images/cats.jpg"



# Get embeddings
factsWithEmbeddings = {fact: get_embedding(fact) for fact in images}
new_fact_embedding = get_embedding(test_image)

if new_fact_embedding:
    # Compute similarities
    similarities = {
        fact: cosineSimilarity(embedding, new_fact_embedding)
        for fact, embedding in factsWithEmbeddings.items() if embedding
    }

    # Sort by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print(f'Similarities for fact: "{test_image}" with:')
    for fact, similarity in sorted_similarities:
        print(f'"{fact}":  "{similarity}"')
else:
    print("Failed to get embedding for the new fact.")