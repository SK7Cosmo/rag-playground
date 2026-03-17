# Importing Packages
import os
import re
import json
import nltk
import warnings
import numpy as np
import pandas as pd
import configparser
import plotly.express as px

warnings.filterwarnings('ignore')

from openai import OpenAI
from numpy.linalg import norm
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from transformers.utils import logging

logging.set_verbosity_error()

# Download Stopwords only if it is not found
try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

# Defining Stopwords
STOPWORDS = set(stopwords.words("english"))

# Reading the config file
creds_config = configparser.ConfigParser()
creds_config.read('config.ini')

# Setting up the OpenRouter Key
keys = dict(creds_config.items('keys'))
openrouter_api_key = keys['openrouter_api_key']
os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

# Initialize OpenRouter client
client = OpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key=os.getenv("OPENROUTER_API_KEY"), )


def remove_stopwords(text):
	"""
	Removes stopwords from the text and return set of unique words
	"""
	words = re.findall(r"\b\w+\b", text.lower())
	return {w for w in words if w not in STOPWORDS}


# Base LLM model
def get_llm_response(user_prompt):
	"""
	Sends a prompt to the OpenAI model through OpenRouter API and returns the response.
	"""

	system_prompt = """You are an helpful AI assistant. You always answer to the user's queries."""

	try:
		response = client.chat.completions.create(
			model="openai/gpt-4o-mini",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt}
			],
			temperature=0.0,  # Factual
			max_tokens=500,  # Maximum length of the generated response.
			top_p=1.0,
			frequency_penalty=0.0,  # To avoid phrase repetitions
			presence_penalty=0.0,  # To avoid topic repetition
		)
		# Extract and return the assistant's message from the response
		return response.choices[0].message.content

	except Exception as e:
		return f"An error occurred: {e}"


def perform_rag_retrieval(query, documents):
	"""
	Gets the document that closely matches for the input query
	Naive way of keyword match technique is used to build a less complex pipeline
	Only the best matching document is used to augment the user prompt
	"""

	# Split the query into lowercase words and store them in a set
	clean_query_words = remove_stopwords(query)
	best_doc_id = None
	best_overlap_score = 0

	for doc_id, doc in documents.items():
		# Remove stopwords from doc content nd thr user query
		# Concat title and content to get efficient match
		# Compare the query words with the document's content words
		# Calculate number of overlapping words of query vs reference doc(s)

		clean_doc_words = remove_stopwords(" ".join(doc["content"]))
		clean_title_words = remove_stopwords(doc['title'])
		doc_master = clean_doc_words.union(clean_title_words)
		overlap_score = len(clean_query_words.intersection(doc_master))

		if overlap_score > best_overlap_score:
			best_overlap_score = overlap_score
			best_doc_id = doc_id

	# Return the best document, or None if nothing matched
	return documents.get(best_doc_id)


def preprocess_string(text):
	"""
	Returns cleaned list of words => remove punctuations around the text
	"""
	words = text.split()
	cleaned_words = [word.lower().strip(".,!?") for word in words]
	return cleaned_words


def build_vocab(docs):
	"""
	Builds vocabulary by assigning unique index to each unigram and bigram
	Creates master vocab by indexing unique words across all the provided docs
	"""
	unique_words = set()  # To assure no duplicate words are added

	for doc in docs:
		words = preprocess_string(doc)
		unique_words.update(set(words))  # Gathering unique unigrams across all the provided docs

		for i in range(len(words)):
			if i < len(words) - 1:
				bigram = words[i] + " " + words[i + 1]
				unique_words.add(bigram)  # Gathering unique bigrams across all the provided docs

	vocab = {word: idx for idx, word in enumerate(sorted(unique_words))}  # Indexing each word
	return vocab


def create_bow_vector(doc, vocab):
	"""
	Creates corresponding vector representation for given input text
	Applies BagOfWords [Frequency Count] on the text using vocabulary for vectorization
	Takes both unigram and bigrams into account
	"""

	# Create a zero vector with the same length as the vocabulary
	vector = np.zeros(len(vocab), dtype=int)
	# Preprocess the input string
	words = preprocess_string(text=doc)

	for i in range(len(words)):
		if words[i] in vocab:
			# Increment the vector slot corresponding to the unigrams and bigraqms, if found in the vocab
			# vocab format => {'word': idx}
			vector[vocab[words[i]]] += 1
			if i < len(words) - 1:
				bigram = words[i] + " " + words[i + 1]
				if bigram in vocab:
					vector[vocab[bigram]] += 1

	return vector


def compute_cosine_similarity(vec_a, vec_b):
	"""
	Compute cosine similarity between two vectors:
	1 means identical direction, 0 means orthogonal.
	"""
	return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def create_sentence_embeddings(sentences):
	"""
	Creates sentence embeddings for corresponding sentence
	Shape (sentence_count, 384)
	"""
	# Load a pre-trained embedding model from Sentence Transformers.
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	embeddings = model.encode(sentences, show_progress_bar=False)  # Sentence => Embedding Vector
	return embeddings


def compute_tsne_embeddings(sentences):
	"""
	Compute and return t-SNE reduced embeddings for the given sentences
	"""
	# Load a pre-trained embedding model from Sentence Transformers.
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	embeddings = model.encode(sentences)

	tsne = TSNE(
		n_components=2,  # No of dims to be reduced to
		random_state=42,
		perplexity=10,  # Neighborhood Count
		max_iter=3000)

	return tsne.fit_transform(embeddings)


def plot_embeddings(reduced_embeddings, sentences, categories):
	# Create the 'plots' folder, if not existing - to store the t-SNE plots
	os.makedirs("plots", exist_ok=True)

	# Prepare dataframe
	df = pd.DataFrame({
		"x": reduced_embeddings[:, 0],
		"y": reduced_embeddings[:, 1],
		"sentence": sentences,
		"category": categories
	})

	fig = px.scatter(
		df,
		x="x", y="y",
		color="category", symbol="category",
		hover_data=["sentence"],
		width=900, height=700)

	fig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))

	fig.update_layout(template="plotly_white", legend_title="Topics")
	fig.update_layout(title="t-SNE Visualization of Sentence Embedding")

	fig.write_html("plots/tsne_plot.html")


def perform_chunk(text, chunk_size=30):
	"""
	Splits the given text into chunks of size 'chunk_size' [word count], preserving sentence boundaries.
	Returns a list of chunk strings.
	"""

	# Split text into semantic sentences
	sentences = re.split(r'[.!?]+', text)

	chunks = []
	current_chunk = ""

	for sentence in sentences:
		if not sentence:
			continue

		# If adding sentence to the chunk exceeds chunk size, save existing chunk
		if len((current_chunk + " " + sentence).split()) > chunk_size:
			chunks.append(current_chunk.strip())
			current_chunk = sentence  # current sentence marks the beginning of new chunk
		else:
			current_chunk += " " + sentence  # Keep adding the sentence to the chunk

	# Add last chunk to the chunk list
	if current_chunk:
		chunks.append(current_chunk.strip())

	return chunks


def load_and_chunk_dataset(data, chunk_size=30):
	"""
	Splits each document of the master dataset into smaller chunks.
	Metadata such as 'doc_id', 'chunk_id' is included with each chunk for easy retrieval.
	"""
	all_chunks = []
	for doc in data:
		doc_id = doc["id"]
		doc_category = doc["category"]
		doc_text = doc["content"]
		doc_chunks = perform_chunk(doc_text, chunk_size)
		for chunk_id, chunk_str in enumerate(doc_chunks):
			all_chunks.append({
				"doc_id": doc_id,
				"category": doc_category,
				"chunk_id": chunk_id,
				"text": chunk_str
			})
	return all_chunks


def build_chroma_collection(chunks, collection_name="rag_collection"):
	"""
	Builds a ChromaDB collection, embedding each chunk using a SentenceTransformer.
	Creates metadata for each chunk for fast retrieval.
	"""
	# Define Sentence Embedding Function
	model_name = 'sentence-transformers/all-MiniLM-L6-v2'
	embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

	# Create the ChromaDB object
	client = Client(Settings())
	collection = client.get_or_create_collection(
		name=collection_name,
		embedding_function=embed_func
	)

	# Preparing the lists of chunk texts, IDs and metadata
	chunk_text_list = [chunk["text"] for chunk in chunks]
	chunk_id_list = [f"chunk_{chunk['doc_id']}_{chunk['chunk_id']}" for chunk in chunks]
	chunk_metadata_list = [
		{
			"doc_id": chunk["doc_id"],
			"chunk_id": chunk["chunk_id"],
			"category": chunk["category"]
		}
		for chunk in chunks
	]

	# Loading the collection
	collection.add(documents=chunk_text_list, metadatas=chunk_metadata_list, ids=chunk_id_list)
	return collection


def delete_documents_with_keyword(collection, keyword):
	"""
	Deletes all documents from the given ChromaDB 'collection' whose text contains 'keyword'.
	"""

	# Get all documents [chunks]
	results = collection.get()

	documents = results["documents"]
	ids = results["ids"]

	ids_to_delete = []

	# Find documents containing the keyword
	for doc, doc_id in zip(documents, ids):
		if keyword.lower() in doc.lower():
			ids_to_delete.append(doc_id)

	# Delete matching documents
	if ids_to_delete:
		collection.delete(ids=ids_to_delete)
		print(f"\nDeleted {len(ids_to_delete)} chunk document(s) containing the word '{keyword}'.")
	else:
		print("\nNo matching documents found.")


if __name__ == "__main__":
	# Define Knowledge Base for document chunking experiment
	with open("data/sk7_knowledge_base3.json", "r") as file_obj:
		dataset = json.load(file_obj)

	master_chunks = load_and_chunk_dataset(data=dataset)
	print("\nLoaded the dataset and created ", len(master_chunks), " chunks from dataset.\n")
