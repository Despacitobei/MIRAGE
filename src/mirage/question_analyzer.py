from typing import Dict, Any
import traceback
from vllm import LLM, SamplingParams

from .prompts import get_question_decomposer_instruction
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

class QuestionAnalyzer:
    """Question analysis agent for decomposing complex questions into sub-questions"""
    
    def __init__(self, llm):
        """Initialize question analyzer
        
        Args:
            llm: vLLM instance for language model inference
        """
        self.llm = llm
    
    def decompose_question(self, question: str) -> Dict[str, Any]:
        """Decompose user question into multiple sub-questions

        Args:
            question: Original user question

        Returns:
            Dict: Dictionary containing sub-questions list and decomposition flag
        """
        config = get_config()

        instruction = get_question_decomposer_instruction(config.search.max_sub_questions)
        prompt = f"{instruction}\n\nPatient query: {question}"

        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.6,
            top_p=0.95,
            top_k=30,
            presence_penalty=1.0,
        )

        outputs = self.llm.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text.strip()

        try:
            import re

            sub_questions_match = re.search(r'--SUB-QUESTIONS--(.*?)--END--', response, re.DOTALL)

            if not sub_questions_match:
                raise ValueError("Cannot extract required sections from output")

            sub_questions_text = sub_questions_match.group(1).strip()
            sub_questions = []

            for line in sub_questions_text.split('\n'):
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    question_text = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question_text:
                        sub_questions.append(question_text)

            if not sub_questions:
                raise ValueError("No valid sub-questions found")

            needs_decomposition = len(sub_questions) > 1

            result = {
                "sub_questions": sub_questions,
                "needs_decomposition": needs_decomposition
            }

            return result

        except Exception as e:
            try:
                import re
                numbered_questions = re.findall(r'^\d+\.\s*(.*?)$', response, re.MULTILINE)

                if numbered_questions:
                    needs_decomposition = len(numbered_questions) > 1
                    return {
                        "sub_questions": numbered_questions,
                        "needs_decomposition": needs_decomposition
                    }
            except:
                pass

            return {
                "sub_questions": [question],
                "needs_decomposition": False
            }

    def update_state_with_decomposition(self, state, decomposition_result):
        """Update state dictionary with decomposition results

        Args:
            state: MultiAgentState instance
            decomposition_result: Decomposition result dictionary

        Returns:
            Updated state
        """
        state.state["decomposition"]["status"] = "completed"

        needs_decomposition = decomposition_result.get("needs_decomposition", False)
        state.state["decomposition"]["needs_decomposition"] = needs_decomposition

        state.state["prompt_type"] = "multi_agent" if needs_decomposition else "single_question"
        state.log(f"Selected prompt type: {state.state['prompt_type']}")

        for sub_q_text in decomposition_result["sub_questions"]:
            sub_q_id = state.add_sub_question(sub_q_text)
            state.log(f"Added sub-question: {sub_q_id} - {sub_q_text}")

        return state
