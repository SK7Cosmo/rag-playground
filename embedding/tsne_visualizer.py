# Importing Packages
import os
import pandas as pd
import plotly.express as px

from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer


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