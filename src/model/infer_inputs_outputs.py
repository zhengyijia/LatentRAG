from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Any
from numpy import ndarray
# from retriever.faiss_retriever_gpu import ChunkItem
from retriever.torch_retriever_gpu import ChunkItem

# common inputs outputs format for iterative retrieval
@dataclass
class IterationRecord:
    query_id: str
    query: str
    golden_answers: List[str] = field(default_factory=list)
    reference_chunk_items: List[ChunkItem] = field(default_factory=list)  # for evaluation only
    reference_subqueries: List[str] = field(default_factory=list)  # for evaluation only
    reference_thoughts: List[str] = field(default_factory=list)  # for evaluation only
    reference_subquery_embeddings: ndarray = None  # for evaluation only
    reference_force_answer: bool = None
    first_reference_topk_chunk_embeddings: ndarray = None  # for evaluation only
    query_embedding: Optional[ndarray] = None
    thoughts: List[str] = field(default_factory=list)
    subqueries: List[str] = field(default_factory=list)
    subquery_embeddings: List[ndarray] = field(default_factory=list)
    chunk_items: List[List[ChunkItem]] = field(default_factory=list)
    hard_neg_chunk_items: List[List[ChunkItem]] = field(default_factory=list)  # for contrastive training
    subanswers: List[str] = field(default_factory=list)
    answer: str = None
    decoded_answer: str = None  # answer decoded by LatentRAG
    force_answer: bool = False
    states: Dict[str, Any] = field(default_factory=dict)  # any additional states to be returned
    reference_em_score: float = None  # EM score from reference model (for comparison)
    reference_cover_em_score: float = None  # Cover EM score from reference model (for comparison)
    reference_answer: str = None  # Answer from reference model (for comparison)
    reference_search_success: float = None  # Search Success score from reference model (for comparison)
    reference_llm_judge_result: Dict[str, Any] = None  # LLM judge result from reference model (for comparison)
    infer_time_each_steps: List[float] = field(default_factory=list)  # time cost for each step
    extra_info: Dict[str, Any] = field(default_factory=dict)  # any extra info
