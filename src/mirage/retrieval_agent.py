from typing import Dict, List, Optional, Any, Union
import re
import time
import sys
import os

from .prompts import get_evidence_retrieval_instruction, get_task_instruction
from .graph_retriever import graph_search, format_graph_results
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

BEGIN_SEARCH_QUERY = "<|KG_QUERY_BEGIN|>"
END_SEARCH_QUERY = "<|KG_QUERY_END|>"
BEGIN_SEARCH_RESULT = "<|KG_RESULT_BEGIN|>"
END_SEARCH_RESULT = "<|KG_RESULT_END|>"

class RetrievalAgent:
    """Retrieval agent for search-based evidence gathering"""
    
    def __init__(self, llm, neo4j_uri=None, neo4j_user=None, neo4j_password=None, tokenizer=None):
        """Initialize retrieval agent"""
        config = get_config()
        
        self.llm = llm
        self.neo4j_uri = neo4j_uri or config.neo4j.uri
        self.neo4j_user = neo4j_user or config.neo4j.user
        self.neo4j_password = neo4j_password or config.neo4j.password
        self.tokenizer = tokenizer
        self.max_search_limit = config.search.max_search_limit
    
    def extract_between(self, text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """Extract content between two tags"""
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
    
    def extract_final_answer(self, text: str) -> str:
        """Extract final answer from model output (after last </think> tag)"""
        think_pattern = re.compile(r"</think>", re.DOTALL)
        all_matches = list(re.finditer(think_pattern, text))
        
        if all_matches:
            last_match = all_matches[-1]
            last_think_end = last_match.end()
            answer = text[last_think_end:].strip()
            return answer
            
        final_answer_pattern = re.compile(r"Final Answer:(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
        match = final_answer_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        return text.strip()
    
    def initialize_sequence(self, state, sub_q_id: str) -> Dict:
        """Initialize retrieval sequence for sub-question"""
        sub_q = state.state["sub_questions"][sub_q_id]
        main_question = state.state["main_question"]
        sub_question = sub_q["text"]
        
        prompt_type = state.state.get("prompt_type", "multi_agent")
        needs_decomposition = state.state["decomposition"].get("needs_decomposition", True)

        instruction = get_evidence_retrieval_instruction(self.max_search_limit)

        if prompt_type == "multi_agent" and needs_decomposition:
            user_prompt = get_task_instruction(sub_question, prompt_type="multi_agent")
        else:
            user_prompt = get_task_instruction(sub_question, prompt_type="single_question")
        
        search_instruction = "\n\nIMPORTANT: You MUST use <think> tags to search the knowledge graph for relevant medical information before answering. Conduct multiple searches using the search tools. After </think>, answer the question directly as a medical expert without mentioning your search process."
        user_prompt += search_instruction
        
        prompt = [{"role": "user", "content": instruction + user_prompt}]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        sequence = {
            "sub_q_id": sub_q_id,
            "prompt": formatted_prompt,
            "output": "",
            "finished": False,
            "history": [],
            "search_count": 0,
            "executed_search_queries": set(),
        }
        
        sub_q["retrieval"]["current_prompt"] = formatted_prompt
        state.update_sub_question_status(sub_q_id, "retrieving")
        
        return sequence
    
    def run_generation(self, sequences: List[Dict]) -> List:
        """Run batch generation for sequences"""
        config = get_config()
        prompts = [s["prompt"] for s in sequences]
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            top_k=config.model.top_k,
            repetition_penalty=config.model.repetition_penalty,
            stop=[END_SEARCH_QUERY, self.tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        
        output_list = self.llm.generate(prompts, sampling_params=sampling_params)
        return output_list
    
    def process_sub_question(self, state, sub_q_id: str, max_turns: int = None, existing_sequence: Dict = None) -> Dict:
        """Process single sub-question retrieval
        
        Args:
            state: State management instance
            sub_q_id: Sub-question ID
            max_turns: Maximum processing turns (optional, from config if not provided)
            existing_sequence: Optional existing sequence to continue processing
        """
        config = get_config()
        max_turns = max_turns or config.search.max_turns
        
        sub_q = state.state["sub_questions"][sub_q_id]
        
        if existing_sequence is not None:
            sequence = existing_sequence
        else:
            sequence = self.initialize_sequence(state, sub_q_id)
        
        turn = 0
        
        while turn < max_turns and not sequence["finished"]:
            turn += 1
            
            outputs = self.run_generation([sequence])
            text = outputs[0].outputs[0].text
            
            sequence["output"] += text
            sequence["prompt"] += text
            sequence["history"].append(text)
            
            sub_q["retrieval"]["current_output"] = sequence["output"]
            
            search_query = self.extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            
            if self.tokenizer.eos_token in text or (not (search_query and sequence["output"].rstrip().endswith(END_SEARCH_QUERY))):
                sequence["finished"] = True
                
                partial_answer = self.extract_final_answer(sequence["output"])
                
                if not partial_answer or len(partial_answer.strip()) < 10:
                    force_answer_prompt = '\n\nPlease answer the question now in one paragraph:\n'
                    sequence["prompt"] += force_answer_prompt
                    sequence["output"] += force_answer_prompt
                    
                    final_outputs = self.run_generation([sequence])
                    final_text = final_outputs[0].outputs[0].text
                    
                    sequence["output"] += final_text
                    sequence["prompt"] += final_text
                    sequence["history"].append(final_text)
                    
                    partial_answer = self.extract_final_answer(sequence["output"])
                
                state.set_partial_answer(sub_q_id, partial_answer)
                state.update_sub_question_status(sub_q_id, "completed")
                sub_q["retrieval"]["partial_answer"] = partial_answer
                
                state.log(f"Sub-question {sub_q_id} completed")
                
                return {
                    "status": "completed",
                    "sub_q_id": sub_q_id,
                    "final_answer": partial_answer,
                    "search_count": sequence["search_count"],
                    "sequence": sequence
                }
            
            if search_query and sequence["output"].rstrip().endswith(END_SEARCH_QUERY):
                if (sequence["search_count"] < self.max_search_limit and 
                    search_query not in sequence["executed_search_queries"]):
                    
                    state.log(f"Sub-question {sub_q_id} - Executing search: {search_query}")
                    
                    try:
                        raw_kg_result = graph_search(
                            search_query, 
                            self.neo4j_uri, 
                            self.neo4j_user, 
                            self.neo4j_password
                        )
                        kg_response = format_graph_results(raw_kg_result)
                        
                        state.add_search_result(sub_q_id, search_query, kg_response)
                        
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}\n{kg_response}\n{END_SEARCH_RESULT}\n\n"
                        sequence["prompt"] += append_text
                        sequence["output"] += append_text
                        sequence["history"].append(append_text)
                        sequence["search_count"] += 1
                        sequence["executed_search_queries"].add(search_query)
                        
                        return {
                            "status": "in_progress",
                            "sub_q_id": sub_q_id,
                            "search_query": search_query,
                            "sequence": sequence
                        }
                        
                    except Exception as e:
                        error_msg = f"Knowledge graph query error: {str(e)}"
                        state.log(f"Sub-question {sub_q_id} - Search error: {str(e)}", "error")
                        
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}\n{error_msg}\n{END_SEARCH_RESULT}\n\n"
                        sequence["prompt"] += append_text
                        sequence["output"] += append_text
                        sequence["history"].append(append_text)
                
                elif sequence["search_count"] >= self.max_search_limit:
                    limit_message = f"\n\n{BEGIN_SEARCH_RESULT}\nYou've reached the maximum search limit ({self.max_search_limit}). Please provide your final answer based on information gathered so far.\n{END_SEARCH_RESULT}\n\n"
                    sequence["prompt"] += limit_message
                    sequence["output"] += limit_message
                    sequence["history"].append(limit_message)
                    
                    state.log(f"Sub-question {sub_q_id} - Reached max search limit")
                
                elif search_query in sequence["executed_search_queries"]:
                    repeat_message = f"\n\n{BEGIN_SEARCH_RESULT}\nYou've already searched for this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n\n"
                    sequence["prompt"] += repeat_message
                    sequence["output"] += repeat_message
                    sequence["history"].append(repeat_message)
                    
                    state.log(f"Sub-question {sub_q_id} - Duplicate search query: {search_query}")
        
        if not sequence["finished"]:
            state.update_sub_question_status(sub_q_id, "failed")
            state.log(f"Sub-question {sub_q_id} exceeded max turns", "warning")
            
            return {
                "status": "failed",
                "sub_q_id": sub_q_id,
                "reason": "Exceeded max turns",
                "sequence": sequence
            }
        
        return {
            "status": "in_progress",
            "sub_q_id": sub_q_id,
            "sequence": sequence
        }
