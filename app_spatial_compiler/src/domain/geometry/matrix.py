from collections.abc import Sequence
from app_spatial_compiler.src.domain.models import SpatialNode

def reconstruct_matrix(nodes: Sequence[SpatialNode], epsilon: float = 5.0) -> str:
    """
    Implements a 2D grid-mapping algorithm to reconstruct multi-line math environments
    such as LaTeX pmatrix.

    1. Projects glyph centroids onto axes to find distinct Rows (R) and Columns (C)
    2. Maps each spatial node to cell (r, c)
    3. Serializes the grid into \\begin{pmatrix} ... \\end{pmatrix}
    """
    if not nodes:
        return ""

    # Sort to determine rows and columns via projection
    sorted_y = sorted(nodes, key=lambda n: n.centroid[1])
    sorted_x = sorted(nodes, key=lambda n: n.centroid[0])

    # Cluster Y-coordinates into distinct rows
    rows = []
    current_row_y = sorted_y[0].centroid[1]
    rows.append(current_row_y)

    for node in sorted_y[1:]:
        if abs(node.centroid[1] - current_row_y) > epsilon:
            current_row_y = node.centroid[1]
            rows.append(current_row_y)

    # Cluster X-coordinates into distinct columns
    cols = []
    current_col_x = sorted_x[0].centroid[0]
    cols.append(current_col_x)

    for node in sorted_x[1:]:
        if abs(node.centroid[0] - current_col_x) > epsilon:
            current_col_x = node.centroid[0]
            cols.append(current_col_x)

    # Initialize empty grid
    grid = [["" for _ in range(len(cols))] for _ in range(len(rows))]

    # Map nodes to the r x c grid
    for node in nodes:
        cx, cy = node.centroid

        # Find nearest row index
        r_idx = min(range(len(rows)), key=lambda i: abs(rows[i] - cy))
        # Find nearest col index
        c_idx = min(range(len(cols)), key=lambda i: abs(cols[i] - cx))

        grid[r_idx][c_idx] += node.char

    # Serialize grid into LaTeX
    latex_lines = []
    for row in grid:
        # Join columns with '&', strip to clean empty concatenations
        latex_lines.append(" & ".join(cell.strip() for cell in row))

    matrix_body = " \\\\ ".join(latex_lines)
    return f"\\begin{{pmatrix}} {matrix_body} \\end{{pmatrix}}"
