from typing import List, Dict, Optional
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class SimilaritySearchResultPlot:
    """Helper class for visualizing similarity search results"""
    
    def __init__(self, 
                 query_vector: List[float], 
                 metadata_vectors: List[List[float]], 
                 labels: Optional[List[str]] = None,
                 similarity_threshold: float = 0.7):
        """Initialize the similarity search result visualization.
        
        Args:
            query_vector: Vector representation of the query
            metadata_vectors: List of vectors to compare against
            labels: Optional labels for metadata vectors
            similarity_threshold: Threshold for considering vectors similar
        """
        self.query_vector = query_vector
        self.metadata_vectors = metadata_vectors
        self.labels = labels or [f"Vector {i+1}" for i in range(len(metadata_vectors))]
        self.similarity_threshold = similarity_threshold
        
        # Calculate similarities between query vector and each metadata vector
        self.similarities = [
            self._calculate_cosine_similarity(query_vector, vec)
            for vec in metadata_vectors
        ]
            
    def _calculate_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension")
            
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def create_visualization(self, 
                           title: str = "Similarity Search Results",
                           height: int = 600,
                           width: int = 1000) -> go.Figure:
        """Create an interactive visualization of similarity scores.
        
        Args:
            title: Title for the visualization
            height: Height of the figure in pixels
            width: Width of the figure in pixels
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=[
                "Query vs. Metadata Vectors",
                "Similarity Scores",
                "Vector Dimensions",
                "Similarity Matrix"
            ],
            specs=[[{"type": "scatter3d"}, {"type": "bar"}],
                  [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # 3D scatter plot of vectors
        self._add_3d_scatter(fig)
        
        # Bar chart of similarity scores
        self._add_similarity_bars(fig)
        
        # Heatmap of vector dimensions
        self._add_dimension_heatmap(fig)
        
        # Similarity matrix heatmap
        self._add_similarity_matrix(fig)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            width=width,
            showlegend=True
        )
        
        return fig
    
    def _add_3d_scatter(self, fig: go.Figure) -> None:
        """Add 3D scatter plot of vectors to the figure."""
        # Use PCA or t-SNE to reduce dimensions if needed
        points = np.array([self.query_vector] + self.metadata_vectors)
        if points.shape[1] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            points = pca.fit_transform(points)
        
        # Plot query vector
        fig.add_trace(
            go.Scatter3d(
                x=[points[0, 0]],
                y=[points[0, 1]],
                z=[points[0, 2]],
                mode='markers',
                name='Query',
                marker=dict(size=8, color='red')
            ),
            row=1, col=1
        )
        
        # Plot metadata vectors
        fig.add_trace(
            go.Scatter3d(
                x=points[1:, 0],
                y=points[1:, 1],
                z=points[1:, 2],
                mode='markers',
                name='Metadata',
                marker=dict(
                    size=6,
                    color=self.similarities,
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=1, col=1
        )
    
    def _add_similarity_bars(self, fig: go.Figure) -> None:
        """Add bar chart of similarity scores to the figure."""
        fig.add_trace(
            go.Bar(
                x=self.labels,
                y=self.similarities,
                marker_color=self.similarities,
                colorscale='Viridis',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add threshold line
        fig.add_hline(
            y=self.similarity_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            row=1, col=2
        )
    
    def _add_dimension_heatmap(self, fig: go.Figure) -> None:
        """Add heatmap of vector dimensions to the figure."""
        all_vectors = np.vstack([self.query_vector] + self.metadata_vectors)
        
        fig.add_trace(
            go.Heatmap(
                z=all_vectors,
                x=[f"Dim {i+1}" for i in range(all_vectors.shape[1])],
                y=['Query'] + self.labels,
                colorscale='Viridis',
                showscale=True
            ),
            row=2, col=1
        )
    
    def _add_similarity_matrix(self, fig: go.Figure) -> None:
        """Add similarity matrix heatmap to the figure."""
        matrix = np.zeros((len(self.metadata_vectors), len(self.metadata_vectors)))
        
        for i in range(len(self.metadata_vectors)):
            for j in range(len(self.metadata_vectors)):
                matrix[i, j] = self._calculate_cosine_similarity(
                    self.metadata_vectors[i],
                    self.metadata_vectors[j]
                )
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=self.labels,
                y=self.labels,
                colorscale='Viridis',
                showscale=True
            ),
            row=2, col=2
        )

def create_similarity_matrix(vectors: List[List[float]], 
                           labels: Optional[List[str]] = None,
                           similarity_threshold: float = 0.7) -> go.Figure:
    """Create a standalone similarity matrix visualization.
    
    Args:
        vectors: List of vectors to compare
        labels: Optional labels for the vectors
        similarity_threshold: Threshold for considering vectors similar
        
    Returns:
        Plotly figure object
    """
    if not labels:
        labels = [f"Vector {i+1}" for i in range(len(vectors))]
    
    # Calculate similarity matrix
    matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            sim = SimilaritySearchResultPlot._calculate_cosine_similarity(
                SimilaritySearchResultPlot,
                vectors[i],
                vectors[j]
            )
            matrix[i, j] = sim
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        zmin=0,
        zmax=1
    ))
    
    # Update layout
    fig.update_layout(
        title="Similarity Matrix",
        xaxis_title="Vectors",
        yaxis_title="Vectors",
        height=600,
        width=800
    )
    
    return fig