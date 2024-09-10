import nltk
import os

# Set the directory for NLTK data
nltk_data_dir = r"./"  # Current directory
nltk.data.path.append(nltk_data_dir)

# Define a list of required NLTK resources
required_resources = ['stopwords', 'punkt', 'brown', 'wordnet', 'omw-1.4',"punkt_tab"]

# Function to check if NLTK resource is downloaded
def check_nltk_resource(resource):
    try:
        nltk.data.find(f'corpora/{resource}')
        return True
    except LookupError:
        return False

# Download required resources if they are not already downloaded
for resource in required_resources:
    if not check_nltk_resource(resource):
        nltk.download(resource, download_dir=nltk_data_dir)
    else:
        pass