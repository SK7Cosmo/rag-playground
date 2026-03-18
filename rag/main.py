# Importing Packages
import json

from ingestion.chunking import load_and_chunk_dataset
from ingestion.chroma_store import build_chroma_collection
from rag.retrieval import retrieve_top_results_by_distance

with open("../data/sk7_knowledge_base3.json", "r") as file_obj:
	dataset = json.load(file_obj)

master_chunks = load_and_chunk_dataset(data=dataset)
print("\nLoaded the dataset and created", len(master_chunks), "chunk(s) from dataset.\n")

collection = build_chroma_collection(chunks=master_chunks, collection_name="rag_collection")
total_chunk_docs = collection.count()
print("\nChromaDB collection created with", total_chunk_docs, "chunk document(s).")

# Search with category filtering
query_input = "Recent advancements in AI and their impact on teaching"
filter_category = "Education".lower()

filter_results = retrieve_top_results_by_distance(
	query=query_input,
	collection=collection,
	categories=[filter_category],
	top_k=3
)

print(f"\nUser Query: {query_input}")
print(f"Filter Category: {filter_category}")
print("\nFiltered Results: ")
if not filter_results:
	print("No matching documents found")
else:
	for res in filter_results:
		print(f"Doc ID: {res['doc_id']}, Category: {res['category']}, Distance: {res['distance']:.4f}")
		print(f"Chunk: {res['content']}\n")