# Importing packages
import json

from utility import get_llm_response, perform_rag_retrieval

# Define Knowledge Base for RAG retrieval
with open("data/sk7_knowledge_base1.json", "r") as file_obj:
	KNOWLEDGE_BASE = json.load(file_obj)


def generate_naive_response(query):
	"""
	Using LLM's pretrained knowledge base to respond to the query
	"""
	prompt = f"Answer directly the following query: {query}"
	return get_llm_response(prompt)


def generate_rag_response(query, document):
	"""
	Using the retrieved custom knowledge to enrich the user prompt
	that helps to customize LLM's response

	If no relevant info found in the knowledge base,
	avoids hallucination and politely refuses to answer
	"""
	if document:
		snippet = f"{document['title']}: {document['content']}"
		prompt = (f"""Using the following information: '{snippet}', answer: {query}.
				Specify that you have made use of SK7 Knowledge Base 1""")
	else:
		prompt = f"""
		No relevant information was retrieved for the question below.
		Politely refuse to answer. Question: {query}
		"""

	return get_llm_response(prompt)


if __name__ == "__main__":
	agent_choice = int(input("""Choose relevant option based on type of agent to be tested: 
			1. Basic Agent
			2. Custom RAG Agent [Keyword Overlap]
			\n Choice: """))
	query = input("\nEnter the Prompt: ")

	if agent_choice == 1:
		# Sample Queries
		"""
		The capital of India is
		Currency followed in New York is
		What day comes after Saturday?
		"""

		print("\nNaive Agent's Response:\n", generate_naive_response(query))

	elif agent_choice == 2:
		# Sample Queries
		"""
		Give an overview on Agentic AI
		What is the workflow for an AI Agentic system
		Name the Primary Components of Agentic AI
		"""

		retrieved_doc = perform_rag_retrieval(query, KNOWLEDGE_BASE)
		print("\nRAG Agent's Response [Keyword Overlap]:\n", generate_rag_response(query, retrieved_doc))

	else:
		print("\nInvalid choice")
