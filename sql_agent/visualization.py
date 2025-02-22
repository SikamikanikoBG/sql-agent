import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SimilaritySearchResultPlot:
    """Helper class for visualizing similarity search results"""
    
    def __init__(self, query_vector: List[float], metadata_vectors: List[List[float]], 
                 similarity_threshold: float = 0.7):
        self.query_vector = query_vector
        self.similarities = []
        self.metadata_vectors = metadata_vectors
        self.similarity_threshold = similarity_threshold
        
        # Calculate similarities between query vector and each metadata vector
        for vec in metadata_vectors:
            sim = self._calculate_cosine_similarity(query_vector, vec)
            self.similarities.append(sim)
            
    def _calculate_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a[i] * b[i] for i in range(len(a)))
        norm_a = math.sqrt(sum(num ** 2 for num in a))
        norm_b = math.sqrt(sum(num ** 2 for num in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def create_visualization(self) -> go.Figure:
        """Create a visualization of similarity scores"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Query Embedding", "Metadata Vectors"])

        # Plot query embedding
        fig.add_trace(go.Scatter(x=[0], y=self.query_vector, name="Query Vector"), row=1, col=1)
        
        # Plot metadata vectors
        for i, vec in enumerate(self.metadata_vectors):
            x = [i+1] * len(vec)
            y = vec
            fig.add_trace(go.Scatter(x=x, y=y, name=f"Metadata {i+1}"), row=1, col=2)

        # Add cosine similarity labels
        for i, sim in enumerate(self.similarities):
            x_coord = i + 2
            y_coord = max(self.query_vector) * 0.9
            
            fig.add_annotation(x=x_coord, y=y_coord, text=f"Similarity: {sim:.4f}", showarrow=False)

        return fig
        
def cosine_similarity_matrix(metadata_vectors: List[List[float]], 
                           similarity_threshold: float = 0.7) -> dict:
    """Calculate and visualize cosine similarity matrix"""
    similarities = []
    
    for i in range(len(metadata_vectors)):
        vec_i = metadata_vectors[i]
        sims = {}
        for j in range(i+1, len(metadata_vectors)):
            vec_j = metadata_vectors[j]
            sim = _calculate_cosine_similarity(vec_i, vec_j)
            sims[j] = sim
            
            if sim >= similarity_threshold:
                sims[j] = {'similarity': sim, 'highlight': True}
        
        similarities.append(sims)
    
    return similarities
