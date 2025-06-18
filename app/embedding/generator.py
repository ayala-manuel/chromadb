# /app/embbeding/generator.py

from sentence_transformers import SentenceTransformer

def load_model():
    """
    Load the SentenceTransformer model.
    """
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the SentenceTransformer model.") from e
    
def get_embedding(text):
    """Embed the input text using the loaded model.

    Args:
        text (str): The text to be embedded.

    Returns:
        list: The embedding of the input text.
    """
    model = load_model()
    try:
        embedding = model.encode(
            text,
            convert_to_numpy=True
            )
        
        return embedding
    
    except Exception as e:
        print(f"Error embedding text: {e}")
        raise RuntimeError("Failed to embed the text.") from e