# Importing Packages
import json
from utility import load_and_chunk_dataset
from utility import build_chroma_collection, delete_documents_with_keyword

# Define Knowledge Base for vector database experiment
with open("data/sk7_knowledge_base3.json", "r") as file_obj:
	dataset = json.load(file_obj)


if __name__ == "__main__":
	master_chunks = load_and_chunk_dataset(data=dataset)
	print("\nLoaded the dataset and created", len(master_chunks), "chunk(s) from dataset.\n")

	collection = build_chroma_collection(chunks=master_chunks, collection_name="rag_collection")
	total_chunk_docs = collection.count()
	print("\nChromaDB collection created with", total_chunk_docs, "chunk document(s).")

	# Now delete all documents containing the keyword "Bananas".
	keyword = "Bananas"
	delete_documents_with_keyword(collection, keyword=keyword)

	final_count = collection.count()
	print(f"\nAfter deleting chunk document(s) with the word '{keyword}', collection has", final_count, "document(s).")

