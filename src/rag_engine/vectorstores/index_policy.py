"""FAISS index auto-selection (plan §9)."""


def choose_faiss_index_type(num_vectors: int, ram_budget_gb: float) -> str:
    if num_vectors < 50_000:
        return "flat"
    if num_vectors < 2_000_000 and ram_budget_gb >= 4:
        return "hnsw"
    if num_vectors < 10_000_000:
        return "ivf_flat"
    return "ivf_pq"
