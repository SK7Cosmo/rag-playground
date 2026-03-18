# Importing Packages
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def build_chroma_collection(chunks, collection_name="rag_collection"):
	"""
	Builds a ChromaDB collection, embedding each chunk using a SentenceTransformer.
	Creates metadata for each chunk for fast retrieval.
	If the collection exists, deletes docs inside it and adds doc from scratch
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

	# Remove existing data (if any) and add fresh documents
	existing_ids = collection.get().get("ids", [])
	if existing_ids:
		collection.delete(ids=existing_ids)

	# Preparing the lists of chunk texts, IDs and metadata
	chunk_text_list = [chunk["content"] for chunk in chunks]
	chunk_id_list = [f"chunk_{chunk['doc_id']}_{chunk['chunk_id']}" for chunk in chunks]
	chunk_metadata_list = [
		{
			"doc_id": chunk["doc_id"],
			"chunk_id": chunk["chunk_id"],
			"category": chunk["category"].lower()
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
