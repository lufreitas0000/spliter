import pytest
from src.domain.models import SpatialNode
from src.domain.geometry.matrix import reconstruct_matrix

def test_reconstruct_2x2_matrix():
    nodes = [
        # Row 1
        SpatialNode(char="a", x0=10, y0=10, x1=12, y1=14),
        SpatialNode(char="b", x0=30, y0=10, x1=32, y1=14),
        # Row 2
        SpatialNode(char="c", x0=10, y0=30, x1=12, y1=34),
        SpatialNode(char="d", x0=30, y0=30, x1=32, y1=34),
    ]

    latex = reconstruct_matrix(nodes, epsilon=5.0)

    # Verify the topological reconstruction
    assert latex == "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"

def test_reconstruct_matrix_with_compound_cells():
    nodes = [
        # Row 1, Col 1: "10"
        SpatialNode(char="1", x0=10, y0=10, x1=12, y1=14),
        SpatialNode(char="0", x0=12, y0=10, x1=14, y1=14),
        # Row 1, Col 2: "2"
        SpatialNode(char="2", x0=30, y0=10, x1=32, y1=14),
    ]

    latex = reconstruct_matrix(nodes, epsilon=5.0)

    assert latex == "\\begin{pmatrix} 10 & 2 \\end{pmatrix}"

def test_reconstruct_empty_matrix():
    assert reconstruct_matrix([]) == ""
