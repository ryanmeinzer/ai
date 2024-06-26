import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

