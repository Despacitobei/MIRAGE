from typing import Dict, List, Any
import re
from vllm import LLM, SamplingParams

from .prompts import get_answer_synthesizer_instruction
from .retrieval_agent import BEGIN_SEARCH_RESULT, END_SEARCH_RESULT
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

class AnswerSynthesizer:
    """Answer synthesis agent for integrating sub-question answers into final answer"""
    
    def __init__(self, llm):
        """Initialize answer synthesizer
        
        Args:
            llm: vLLM instance for language model inference
        """
        self.llm = llm
    
    def collect_sub_question_answers(self, state) -> Dict[str, Dict]:
        """Collect all sub-question answers

        Args:
            state: State management instance

        Returns:
            Dict: Mapping from sub-question ID to answer
        """
        sub_question_answers = {}
        completed_count = 0
        total_count = 0

        for sub_q_id, sub_q in state.state["sub_questions"].items():
            total_count += 1
            if sub_q["status"] == "completed":
                answer = sub_q["retrieval"]["partial_answer"]
                if answer and len(answer.strip()) > 10:
                    completed_count += 1
                    sub_question_answers[sub_q_id] = {
                        "text": sub_q["text"],
                        "answer": answer
                    }

        return sub_question_answers

    def collect_valid_evidences(self, state) -> List[Dict]:
        """Collect all valid knowledge graph evidences

        Args:
            state: State management instance

        Returns:
            List[Dict]: List of valid evidences
        """
        evidences = state.get_all_valid_evidences()
        return evidences
    
    def synthesize_answer(self, state) -> Dict[str, Any]:
        """Synthesize final answer from sub-question answers
        
        Args:
            state: State management instance
            
        Returns:
            Dict: Synthesis result with final_answer and metadata
        """
        config = get_config()
        
        sub_answers = self.collect_sub_question_answers(state)
        valid_evidences = self.collect_valid_evidences(state)

        state.state["synthesis"]["status"] = "in_progress"
        state.state["synthesis"]["input_summary"] = {
            "sub_answers": sub_answers,
            "valid_evidences": valid_evidences
        }
        
        instruction = get_answer_synthesizer_instruction()
        prompt = f"{instruction}\n\n"

        if sub_answers:
            prompt += "SUB-QUESTION ANSWERS:\n"
            for info in sub_answers.values():
                prompt += f"SUB-QUESTION: {info['text']}\n"
                prompt += f"ANSWER: {info['answer']}\n\n"

        if valid_evidences:
            prompt += "KNOWLEDGE GRAPH EVIDENCE:\n"
            for i, evidence in enumerate(valid_evidences, 1):
                prompt += f"{i}. Query: {evidence['query']}\n"
                prompt += f"   Result: {evidence['result']}\n\n"

        prompt += f"Please carefully analyze the following medical question and provide a professional diagnosis. After your reasoning, address these three key aspects in detail:\n"
        prompt += "1. What disease is the patient most likely to have based on the symptoms and clinical presentation?\n"
        prompt += "2. What tests should be performed to confirm the diagnosis?\n"
        prompt += "3. What medications or treatments are recommended to manage or cure the disease?\n\n"
        prompt += f"PATIENT'S ORIGINAL QUESTION:\n{state.state['main_question']}\n\n"
        prompt += "IMPORTANT: The information above is external medical knowledge for your reference. Your task is to answer the patient's original question. Base your answer ONLY on what the patient actually said, not on hypothetical scenarios or additional conditions mentioned in the reference material.\n"
        prompt += "First use <think> tags to review the information above. Then provide your final answer in <answer> tags.\n"
        prompt += "Answer in 1-2 paragraphs without bullet points, numbered lists, or special formatting. Write as a real doctor would.\n"
        
        sampling_params = SamplingParams(
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            top_k=config.model.top_k,
            repetition_penalty=config.model.repetition_penalty,
        )
        
        state.log("Generating final answer...")
        outputs = self.llm.generate(prompt, sampling_params=sampling_params)
        final_answer = outputs[0].outputs[0].text.strip()
        
        cleaned_answer = self.clean_final_answer(final_answer)

        state.set_final_answer(cleaned_answer)
        state.state["synthesis"]["status"] = "completed"

        state.log("Final answer generation completed")

        return {
            "status": "success",
            "final_answer": cleaned_answer,
            "sub_questions_used": list(sub_answers.keys()),
            "synthesis_metadata": {
                "sub_questions_count": len(sub_answers),
                "answer_length": len(cleaned_answer)
            }
        }
    
    def clean_final_answer(self, answer: str) -> str:
        """Clean final answer by extracting medical response after </think>

        Args:
            answer: Raw answer

        Returns:
            str: Cleaned answer
        """
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        all_answer_matches = list(re.finditer(answer_pattern, answer))

        if all_answer_matches:
            last_answer_match = all_answer_matches[-1]
            cleaned = last_answer_match.group(1).strip()

            cleaned = re.sub(r'^\s*\n+', '', cleaned)
            cleaned = cleaned.strip()

            if cleaned and len(cleaned) > 20:
                return cleaned

        think_pattern = re.compile(r'</think>', re.DOTALL)
        all_think_matches = list(re.finditer(think_pattern, answer))

        if all_think_matches:
            last_think_match = all_think_matches[-1]
            last_think_end = last_think_match.end()
            cleaned = answer[last_think_end:].strip()

            cleaned = re.sub(r'<[^>]+>', '', cleaned)
            cleaned = re.sub(r'^\s*\n+', '', cleaned)
            cleaned = cleaned.strip()

            if cleaned and len(cleaned) > 20:
                return cleaned

        cleaned = answer

        prefixes = [
            "Here's my final answer:",
            "Final answer:",
            "Based on the information provided,",
            "Based on the sub-question answers,",
            "Here is the comprehensive answer:",
            "Synthesized answer:"
        ]

        for prefix in prefixes:
            if cleaned.lstrip().startswith(prefix):
                cleaned = cleaned.lstrip()[len(prefix):].strip()

        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'^\s*\n+', '', cleaned)
        cleaned = cleaned.strip()

        return cleaned
