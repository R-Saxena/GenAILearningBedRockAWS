from similiarity import cosineSimilarity
import boto3
import json

client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

facts = [
    "A group of flamingos is called a 'flamboyance.'",
    "Honey never spoils; archaeologists have found pots of honey in ancient Egyptian tombs that are still edible.",
    "Bananas are berries, but strawberries aren't.",
    "Octopuses have three hearts and blue blood.",
    "A day on Venus is longer than a year on Venus."
]

new_fact = "The first moon landing was in 1969."

def get_embedding(input_str: str):
    try:
        response = client.invoke_model(
            body=json.dumps({"inputText": input_str}),
            modelId="amazon.titan-embed-text-v1",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding", [])
    except Exception as e:
        print(f"Error fetching embedding for '{input_str}': {e}")
        return None

# Get embeddings
factsWithEmbeddings = {fact: get_embedding(fact) for fact in facts}
new_fact_embedding = get_embedding(new_fact)

if new_fact_embedding:
    # Compute similarities
    similarities = {
        fact: cosineSimilarity(embedding, new_fact_embedding)
        for fact, embedding in factsWithEmbeddings.items() if embedding
    }

    # Sort by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print(f'Similarities for fact: "{new_fact}" with:')
    for fact, similarity in sorted_similarities:
        print(f'"{fact}":  "{similarity}"')
else:
    print("Failed to get embedding for the new fact.")
