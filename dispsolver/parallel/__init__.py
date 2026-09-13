"""
dispsolver/parallel
===================
MPI and Distributed-Memory Parallel Computing module for WHT_DispFoldSolver.
Provides Domain Decomposition, Subdomain Mesh Partitioning, and Distributed Assembly.
"""

from .partitioner import DomainPartitioner, Subdomain

__all__ = [
    "DomainPartitioner",
    "Subdomain",
]
