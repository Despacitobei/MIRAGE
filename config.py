import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str = "/path/to/your/QwQ-32B-AWQ"  # Path to your LLM model
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.95
    max_tokens: int = 32768
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05

@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str = "bolt://localhost:7687"  # Your Neo4j server URI
    user: str = "neo4j"
    password: str = "your-password"  # Your Neo4j password

@dataclass
class SearchConfig:
    """Search configuration"""
    max_search_limit: int = 5
    max_sub_questions: int = 4
    max_retries: int = 3
    max_turns: int = 10
    entity_match_threshold: float = 0.7

@dataclass
class PathConfig:
    """Path configuration"""
    base_output_dir: str = "./outputs"
    triples_file: str = "./data/chatdoctor5k/train.txt"
    entity_embeddings_path: str = "./data/chatdoctor5k/entity_embeddings.pkl"
    sentence_transformer_path: str = "/path/to/your/all-mpnet-base-v2"  # Path to sentence transformer model
    
    @property
    def results_dir(self) -> str:
        """Results directory"""
        return os.path.join(self.base_output_dir, "results")
    
    @property  
    def logs_dir(self) -> str:
        """Logs directory"""
        return os.path.join(self.base_output_dir, "logs")

@dataclass
class DataConfig:
    """Dataset configuration"""
    default_dataset_path: str = "./data/chatdoctor5k/NER_chatgpt.json"
    subset_num: int = -1  # -1 for all data

@dataclass
class AppConfig:
    """Application configuration"""
    model: ModelConfig
    neo4j: Neo4jConfig
    search: SearchConfig
    paths: PathConfig
    data: DataConfig
    
    def __init__(self):
        self.model = ModelConfig()
        self.neo4j = Neo4jConfig()
        self.search = SearchConfig()
        self.paths = PathConfig()
        self.data = DataConfig()
        os.makedirs(self.paths.results_dir, exist_ok=True)
        os.makedirs(self.paths.logs_dir, exist_ok=True)

config = AppConfig()

def get_config() -> AppConfig:
    """Get global config instance"""
    return config

def update_config_from_args(args):
    """Update config from command line arguments"""
    if hasattr(args, 'model_path') and args.model_path:
        config.model.model_path = args.model_path
    
    if hasattr(args, 'neo4j_uri') and args.neo4j_uri:
        config.neo4j.uri = args.neo4j_uri
    
    if hasattr(args, 'neo4j_user') and args.neo4j_user:
        config.neo4j.user = args.neo4j_user
        
    if hasattr(args, 'neo4j_password') and args.neo4j_password:
        config.neo4j.password = args.neo4j_password
    
    if hasattr(args, 'max_sub_questions') and args.max_sub_questions:
        config.search.max_sub_questions = args.max_sub_questions
        
    if hasattr(args, 'max_retries') and args.max_retries:
        config.search.max_retries = args.max_retries
        
    if hasattr(args, 'dataset_path') and args.dataset_path:
        config.data.default_dataset_path = args.dataset_path
        
    if hasattr(args, 'subset_num') and args.subset_num:
        config.data.subset_num = args.subset_num
