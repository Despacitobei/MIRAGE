from typing import Dict, List, Any, Optional
import time
import json
import os

from .agent_state import MultiAgentState
from .question_analyzer import QuestionAnalyzer
from .retrieval_agent import RetrievalAgent
from .synthesizer import AnswerSynthesizer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

class MultiAgentCoordinator:
    """Multi-agent coordinator managing agent workflow"""
    
    def __init__(self, llm, tokenizer, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        """Initialize coordinator
        
        Args:
            llm: vLLM instance
            tokenizer: Tokenizer
            neo4j_uri: Neo4j database URI (optional, from config if not provided)
            neo4j_user: Neo4j username (optional, from config if not provided)
            neo4j_password: Neo4j password (optional, from config if not provided)
        """
        config = get_config()
        
        self.llm = llm
        self.tokenizer = tokenizer
        self.neo4j_uri = neo4j_uri or config.neo4j.uri
        self.neo4j_user = neo4j_user or config.neo4j.user
        self.neo4j_password = neo4j_password or config.neo4j.password
        
        self.results_dir = config.paths.results_dir
        self.logs_dir = config.paths.logs_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.results_file = os.path.join(self.results_dir, f"result_{timestamp}.jsonl")
        self.log_file = os.path.join(self.logs_dir, f"log_{timestamp}.jsonl")
        self.results = []
        self.logs = []
        
        self.question_analyzer = QuestionAnalyzer(self.llm)
        self.retrieval_agent = RetrievalAgent(self.llm, self.neo4j_uri, 
                                             self.neo4j_user, self.neo4j_password, self.tokenizer)
        self.synthesizer = AnswerSynthesizer(self.llm)
    
    def save_result(self, result_data):
        """Save simplified results to file (JSONL format)"""
        self.results.append(result_data)
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
        print(f"[Coordinator] Results saved to: {self.results_file}")
    
    def save_log(self, log_data):
        """Save detailed logs to file (JSONL format)"""
        self.logs.append(log_data)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        print(f"[Coordinator] Log saved to: {self.log_file}")
    
    def process_question(self, question: str, max_sub_questions: int = 4, max_retries: int = 3) -> Dict:
        """Process user question
        
        Args:
            question: User question
            max_sub_questions: Maximum number of sub-questions
            max_retries: Maximum retry attempts
            
        Returns:
            Dict: Processing results
        """
        state = MultiAgentState(question)
        start_time = time.time()
        
        # Phase 1: Question decomposition
        decomposition_result = self.question_analyzer.decompose_question(question)
        self.question_analyzer.update_state_with_decomposition(state, decomposition_result)
        
        needs_decomposition = state.state["decomposition"].get("needs_decomposition", True)
        
        sub_q_ids = list(state.state["sub_questions"].keys())
        if len(sub_q_ids) > max_sub_questions:
            sub_q_ids = sub_q_ids[:max_sub_questions]
        
        # Phase 2: Sub-question processing
        all_search_queries = set()
        
        for i, sub_q_id in enumerate(sub_q_ids):
            sub_q = state.state["sub_questions"][sub_q_id]
            retry_count = 0
            sequence = None
            
            while state.state["sub_questions"][sub_q_id]["status"] not in ["completed", "failed"]:
                retrieval_result = self.retrieval_agent.process_sub_question(
                    state, sub_q_id, existing_sequence=sequence
                )
                
                if "sequence" in retrieval_result:
                    sequence = retrieval_result["sequence"]
                    
                if "search_query" in retrieval_result:
                    all_search_queries.add(retrieval_result["search_query"])
                
                if retrieval_result["status"] in ["completed", "failed"]:
                    break
            
                if retrieval_result["status"] == "in_progress":
                    continue
                
                retry_count += 1
                if retry_count >= max_retries:
                    state.update_sub_question_status(sub_q_id, "failed")
                    break
        
        # Phase 3: Answer synthesis
        sub_answers = self.synthesizer.collect_sub_question_answers(state)
        synthesis_result = self.synthesizer.synthesize_answer(state)
        final_answer = synthesis_result["final_answer"]
        
        processing_time = time.time() - start_time
        
        result = {
            "question": question,
            "final_answer": final_answer,
            "sub_questions": [
                {
                    "id": sub_q_id,
                    "text": state.state["sub_questions"][sub_q_id]["text"],
                    "status": state.state["sub_questions"][sub_q_id]["status"],
                    "partial_answer": state.state["sub_questions"][sub_q_id]["retrieval"]["partial_answer"]
                }
                for sub_q_id in state.state["sub_questions"]
            ],
            "metrics": {
                "processing_time": processing_time,
                "total_searches": state.state["metrics"]["total_searches"],
                "successful_validations": state.state["metrics"]["successful_validations"],
                "failed_validations": state.state["metrics"]["failed_validations"]
            },
            "state": state.get_state()
        }
        
        simplified_result = {
            "question": question,
            "sub_questions": decomposition_result["sub_questions"],
            "needs_decomposition": needs_decomposition,
            "executed_search_queries": list(all_search_queries),
            "final_answer": final_answer,
            "search_count": len(all_search_queries),
            "processing_time": processing_time
        }
        
        self.save_result(simplified_result)
        
        detailed_log = {
            "question": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_answer": final_answer,
            "decomposition": {
                "sub_questions": decomposition_result["sub_questions"],
                "needs_decomposition": needs_decomposition
            },
            "sub_questions_details": [
                {
                    "id": sub_q_id,
                    "text": state.state["sub_questions"][sub_q_id]["text"],
                    "status": state.state["sub_questions"][sub_q_id]["status"],
                    "retrieval_output": state.state["sub_questions"][sub_q_id]["retrieval"]["current_output"],
                    "partial_answer": state.state["sub_questions"][sub_q_id]["retrieval"]["partial_answer"],
                    "search_history": [
                        {"query": search["query"], "result": search["result"]}
                        for search in state.state["sub_questions"][sub_q_id]["retrieval"]["search_history"]
                    ]
                }
                for sub_q_id in state.state["sub_questions"]
            ],
            "metrics": {
                "processing_time": processing_time,
                "total_searches": state.state["metrics"]["total_searches"],
                "search_queries": list(all_search_queries)
            }
        }
        
        self.save_log(detailed_log)
        return result
