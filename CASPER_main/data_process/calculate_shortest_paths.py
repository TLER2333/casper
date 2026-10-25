"""Convert a directed adjacency CSV into the shortest-path matrix used by CASPER."""
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adjacency = pd.read_csv(args.adjacency, index_col=0)
    if list(adjacency.index) != list(adjacency.columns):
        raise ValueError("adjacency rows and columns must use the same gene order")

    distances = dijkstra(
        csgraph=csr_matrix(adjacency.to_numpy()),
        directed=True,
        unweighted=True,
    )
    distances[np.isinf(distances)] = 0
    pd.DataFrame(
        distances, index=adjacency.index, columns=adjacency.columns
    ).to_csv(args.output)


if __name__ == "__main__":
    main()
