from PIL import Image
import requests
from io import BytesIO

import streamlit as st 
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# env variables
API_KEY = st.secrets['PINECONE_API_KEY']
DB = st.secrets['PINECONE_DB']



def url_to_bytes(url: str) -> BytesIO:
    """
    Convert a URL to bytes using a GET request.

    Args:
        url (str): The URL to convert to bytes.

    Returns:
        BytesIO: A BytesIO object containing the content of the URL.
    """
    response = requests.get(url)
    return BytesIO(response.content)

class ProductNode:
    """
        Initializes the object with the provided id, score, image, name, and metadata.

        Args:
            id (str): The identifier for the object.
            score (str): The score associated with the object.
            image (str): The image associated with the object.
            name (str): The name associated with the object.
            metadata (dict): Additional metadata for the object.

        Returns:
            None
    """
    def __init__(self, id: str, score: str, image: str, name: str,metadata: dict):
        self.id = id
        self.score = score 
        self.image = metadata['image']
        self.name = metadata['name']


#init pinecone client 
pc = Pinecone(api_key=API_KEY)
index = pc.Index(DB)

st.header("Indus Valley Search Engine")

#loading the model 
try:
    model = SentenceTransformer('./all-mpnet-base-v2',device='cpu')
except Exception as e:
    print(e)

user_query = st.text_input(label='Search for the product you need ')
top_k = st.number_input(label='top_k', min_value=1, max_value=10, step=1, value=5)

with st.spinner("getting products.... "):
    user_query_embedding = model.encode(user_query).tolist()
    products = index.query(vector = user_query_embedding, 
                           top_k = top_k,
                           include_metadata=True)

    for product in products['matches']:
        prd_node = ProductNode(**product)
        st.markdown(f"**{prd_node.name}**")
        st.markdown(f"**{prd_node.score}**")
        st.image(Image.open(url_to_bytes(prd_node.image)))
        st.divider()








