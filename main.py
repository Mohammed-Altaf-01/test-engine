from PIL import Image
import requests
from io import BytesIO

# example code to read the image from the url and display
# url ="https://cdn.shopify.com/s/files/1/0594/7251/1153/products/Artboard1_2.jpg?v=1643873829"
# response = requests.get(url)
# Image.open(BytesIO(response.content))



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


#init pinecone client 
pc = Pinecone(api_key=API_KEY)
index = pc.Index("test-search-optim")

st.header("Indus Valley Search Engine")

#loading the model 
try:
    model = SentenceTransformer('./all-mpnet-base-v2',device='cpu')
except error as e:
    print(e)





