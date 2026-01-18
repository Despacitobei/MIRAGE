import uuid
import time
from typing import Dict, List

class MultiAgentState:
    """Multi-agent state management"""
    
    def __init__(self, question: str):
        """Initialize state
        
        Args:
            question: Main question from user
        """
        self.state = {
            "main_question": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": str(uuid.uuid4()),
            
            "decomposition": {
                "sub_questions": [],
                "status": "pending", 
                "error": None
            },
            
            "sub_questions": {},
            
            "synthesis": {
                "status": "pending",
                "input_summary": {},
                "conflicts": [],
                "integration_notes": "",
                "final_answer": "",
            },
            
            "metrics": {
                "start_time": time.time(),
                "end_time": 0,
                "total_tokens": 0,
                "total_searches": 0,
                "successful_validations": 0,
                "failed_validations": 0,
                "agent_transitions": [],
            },
            
            "debug": {
                "logs": [],
                "warnings": [],
                "errors": [],
                "agent_states": {}
            }
        }
    
    def add_sub_question(self, sub_question: str) -> str:
        """Add sub-question and return its ID

        Args:
            sub_question: Sub-question text

        Returns:
            Sub-question ID
        """
        sub_q_id = f"sub_q_{len(self.state['sub_questions']) + 1}"

        self.state["sub_questions"][sub_q_id] = {
            "text": sub_question,
            "status": "pending",
            "attempts": 0,
            "max_attempts": 3,
            
            "retrieval": {
                "current_prompt": "",
                "current_output": "",
                "search_count": 0,
                "executed_queries": set(),
                "search_history": [],
                "last_search_result": None,
                "reasoning_steps": [],
                "partial_answer": "",
                "valid_evidences": [],
            },
            
            "validation": {
                "validated_searches": [],
                "rejected_searches": [],
                "validation_history": [],
                "evidence_quality": 0,
                "needs_retry": False,
                "retry_strategy": None,
            }
        }
        
        self.state["decomposition"]["sub_questions"].append(sub_question)
        return sub_q_id
    
    def update_sub_question_status(self, sub_q_id: str, status: str) -> None:
        """Update sub-question status"""
        if sub_q_id in self.state["sub_questions"]:
            self.state["sub_questions"][sub_q_id]["status"] = status
            self.log(f"Sub-question {sub_q_id} status changed to: {status}")
    
    def add_search_result(self, sub_q_id: str, query: str, result: str) -> None:
        """Add search result"""
        if sub_q_id in self.state["sub_questions"]:
            sub_q = self.state["sub_questions"][sub_q_id]

            sub_q["retrieval"]["search_count"] += 1
            self.state["metrics"]["total_searches"] += 1
            sub_q["retrieval"]["executed_queries"].add(query)

            search_entry = {
                "query": query,
                "result": result,
                "timestamp": time.time()
            }
            sub_q["retrieval"]["search_history"].append(search_entry)
            sub_q["retrieval"]["last_search_result"] = search_entry

            if self._is_valid_evidence(result):
                if not self._is_duplicate_evidence(sub_q["retrieval"]["valid_evidences"], result):
                    evidence_entry = {
                        "query": query,
                        "result": result,
                        "timestamp": time.time()
                    }
                    sub_q["retrieval"]["valid_evidences"].append(evidence_entry)

    def _is_valid_evidence(self, result: str) -> bool:
        """Check if search result is valid evidence"""
        error_indicators = [
            "You've reached the maximum search limit",
            "No matches found",
            "Query Error",
            "Error in graph_search",
            "No paths found",
            "No neighbors found"
        ]

        result_lower = result.lower()
        for error in error_indicators:
            if error.lower() in result_lower:
                return False

        if not result.strip():
            return False

        return True

    def _is_duplicate_evidence(self, existing_evidences: List[Dict], new_result: str) -> bool:
        """Check if new search result duplicates existing evidence"""
        normalized_new = new_result.strip().lower()

        for evidence in existing_evidences:
            existing_result = evidence["result"].strip().lower()

            if normalized_new == existing_result:
                return True

            if (normalized_new in existing_result and len(normalized_new) > 20) or \
               (existing_result in normalized_new and len(existing_result) > 20):
                return True

        return False

    def add_validation_result(self, sub_q_id: str, validation: Dict, is_valid: bool) -> None:
        """Add validation result"""
        if sub_q_id in self.state["sub_questions"]:
            sub_q = self.state["sub_questions"][sub_q_id]
            last_search = sub_q["retrieval"]["last_search_result"]
            
            if last_search:
                validated_search = {
                    "query": last_search["query"],
                    "result": last_search["result"],
                    "validation": validation,
                    "is_valid": is_valid,
                    "timestamp": time.time()
                }
                
                sub_q["validation"]["validation_history"].append(validated_search)
                
                if is_valid:
                    sub_q["validation"]["validated_searches"].append(validated_search)
                    self.state["metrics"]["successful_validations"] += 1
                else:
                    sub_q["validation"]["rejected_searches"].append(validated_search)
                    sub_q["validation"]["needs_retry"] = True
                    self.state["metrics"]["failed_validations"] += 1
    
    def set_partial_answer(self, sub_q_id: str, answer: str) -> None:
        """Set partial answer for sub-question"""
        if sub_q_id in self.state["sub_questions"]:
            self.state["sub_questions"][sub_q_id]["retrieval"]["partial_answer"] = answer
    
    def set_final_answer(self, answer: str) -> None:
        """Set final answer"""
        self.state["synthesis"]["final_answer"] = answer
        self.state["synthesis"]["status"] = "completed"
        self.state["metrics"]["end_time"] = time.time()
        
        duration = self.state["metrics"]["end_time"] - self.state["metrics"]["start_time"]
        self.log(f"Completed in {duration:.2f}s")
    
    def log(self, message: str, level: str = "info") -> None:
        """Add log entry"""
        log_entry = {
            "message": message,
            "timestamp": time.time(),
            "level": level
        }
        
        self.state["debug"]["logs"].append(log_entry)
        
        if level == "warning":
            self.state["debug"]["warnings"].append(log_entry)
        elif level == "error":
            self.state["debug"]["errors"].append(log_entry)
    

    
    def all_sub_questions_completed(self) -> bool:
        """Check if all sub-questions are completed"""
        return all(
            sub_q["status"] in ["completed", "failed"]
            for sub_q in self.state["sub_questions"].values()
        )
    
    def get_all_valid_evidences(self) -> List[Dict]:
        """Get all valid evidences from sub-questions"""
        all_evidences = []
        for sub_q_id, sub_q in self.state["sub_questions"].items():
            for evidence in sub_q["retrieval"]["valid_evidences"]:
                evidence_with_context = {
                    "sub_question_id": sub_q_id,
                    "sub_question_text": sub_q["text"],
                    "query": evidence["query"],
                    "result": evidence["result"],
                    "timestamp": evidence["timestamp"]
                }
                all_evidences.append(evidence_with_context)
        return all_evidences

    def get_state(self) -> Dict:
        """Get complete state dictionary"""
        return self.state
