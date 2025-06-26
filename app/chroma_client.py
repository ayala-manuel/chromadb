# app/chroma_client.py

from chromadb import PersistentClient
import uuid

chroma_client = PersistentClient(
    path="vectorstore/"
)

def create_collection(name:str, embed_description : str):
    """
    Create a new collection in the ChromaDB client.
    
    Args:
        name (str): The name of the collection to create.
            
    Returns:
        Collection: The created collection object.
    """
    return chroma_client.create_collection(
        name=name,
        metadata={
            "uuid": str(uuid.uuid4()),
            "description": embed_description
        }
    )

def delete_collection(name: str):
    """
    Delete a collection from the ChromaDB client.
    
    Args:
        name (str): The name of the collection to delete.
        
    Returns:
        str: A message indicating the result of the deletion.
    """
    try:
        chroma_client.delete_collection(name=name)
        return f"Collection {name} deleted successfully."
    except Exception as e:
        return f"Error deleting collection {name}: {str(e)}"

def upload_data(collection_name: str, data : dict):
    """
    Load data into a specified collection in the ChromaDB client.
    
    Args:
        data (dict): The data to load into the collection.
        collection_name (str): The name of the collection to load data into.
        
    Returns:
        Collection: The collection object with the loaded data.

    Example data format:
    {
        "documents": ["Document 1", "Document 2"],
        "metadata": [{"key": "value"}, {"key": "value"}]
    }
    Note: The lengths of documents, metadata, and embeddings must match.
    """
    from app.embedding.generator import get_embedding
    try:
        print(f"Loading data into collection: {collection_name}")
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Collection {collection_name} found.")

        documents = data.get("documents", [])
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        metadata = data.get("metadata", [{}] * len(documents))
        embeddings = [get_embedding(doc) for doc in documents]
        print(f"Embedding {len(embeddings)} documents.")

        if not (len(documents) == len(metadata) == len(embeddings)):
            raise ValueError("Documents, metadata, and embeddings must be of the same length.")

        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadata,
            embeddings=embeddings
        )

        return collection
    except Exception as e:
        return f"Collection {collection_name} not found : {str(e)}"
    
def retrieve_information(query: str, collection_name : str, n_results: int =5):
    """
    Retrieve information from the ChromaDB client based on a query.
    
    Args:
        query (str): The query string to search for.
        n_results (int): The number of results to return.
        
    Returns:
        list: A list of documents matching the query.
    """
    from app.embedding.generator import get_embedding
    embed_query = get_embedding(query)
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings = [embed_query],
        n_results = n_results
    )

    return results

def return_collection_names():
    """
    Return the names of all collections in the ChromaDB client.
    
    Returns:
        list: A list of collection names.
    """
    collections = chroma_client.list_collections()

    return [
        {
            "collection_name": collection.name,
            "metadata": {
                "uuid": collection.metadata.get("uuid", ""),
                "description": collection.metadata.get("description", "")
            }
        }
        for collection in collections
    ]

