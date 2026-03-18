# Importing packages
import json

from utility import get_llm_response
from utility import retrieve_top_result_by_keyword_overlap, retrieve_top_results_by_distance
from utility import load_and_chunk_dataset, build_chroma_collection

# Define Knowledge Base for RAG retrieval
with open("data/sk7_knowledge_base1.json", "r") as file_obj:
	KNOWLEDGE_BASE = json.load(file_obj)

# Define Knowledge Base for vector database experiment
with open("data/sk7_knowledge_base3.json", "r") as file_obj:
	dataset = json.load(file_obj)


def generate_naive_response(query):
	"""
	Using LLM's pretrained knowledge base to respond to the query
	"""
	prompt = f"Answer directly the following query: {query}"
	return get_llm_response(prompt)


def generate_rag_response(query, rag_content):
	"""
	Using the custom knowledge base to enrich the user prompt
	and customize the LLM's response

	If no info relevant to the query found in the knowledge base,
	avoids hallucination and politely refuses to answer
	"""
	if rag_content:
		prompt = f"Question: {query}\nAnswer using only the following context:\n"
		for fact in rag_content:
			prompt += f"- {fact}\n"
		prompt += "Also, Specify that you have made use of preconfigured Knowledge Base in new line"
		prompt += "\nAnswer: "

	else:
		prompt = f"""
		No relevant information was retrieved for the question below.
		Politely refuse to answer. Question: {query}
		"""

	return get_llm_response(prompt)


if __name__ == "__main__":
	master_chunks = load_and_chunk_dataset(data=dataset)
	print("\nLoaded the dataset and created", len(master_chunks), "chunk(s) from dataset.\n")

	collection = build_chroma_collection(chunks=master_chunks, collection_name="rag_collection")
	total_chunk_docs = collection.count()
	print("\nChromaDB collection created with", total_chunk_docs, "chunk document(s).")

	agent_choice = int(input("""\nChoose relevant option based on type of agent to be tested: 
			1. Basic Agent
			2. Custom RAG Agent - Keyword Overlap based [JSON Source]
			3. Custom RAG Agent - Distance based [ChromaDB Source]
			\nChoice: """))
	query = input("\nEnter the Prompt: ")

	if agent_choice == 1:
		# Sample Queries
		"""
		The capital of India is
		Currency followed in New York is
		What day comes after Saturday?
		"""

		print("\nNaive Agent's Response:\n\n", generate_naive_response(query))

	elif agent_choice == 2:
		# Sample Queries
		"""
		Give an overview on Agentic AI
		What is the workflow for an AI Agentic system
		Name the Primary Components of Agentic AI
		"""

		retrieved_doc = retrieve_top_result_by_keyword_overlap(query, KNOWLEDGE_BASE)
		if retrieved_doc:
			rag_content = retrieved_doc["content"]
		else:
			rag_content = None
		print("\nRAG Agent's Response [Keyword Overlap based]:\n\n", generate_rag_response(query=query, rag_content=rag_content))

	elif agent_choice == 3:
		# Sample Queries
		"""
		What are some recent technological breakthroughs?
		"""
		rag_content = []

		retrieved_chunks = retrieve_top_results_by_distance(query=query, collection=collection, top_k=3, distance_threshold=1.0)

		for chunk in retrieved_chunks:
			rag_content.append(chunk['content'])
		print("\nRAG Agent's Response [Distance based]:\n\n", generate_rag_response(query=query, rag_content=rag_content))

	else:
		print("\nInvalid choice")
