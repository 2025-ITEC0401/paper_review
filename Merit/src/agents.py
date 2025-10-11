# src/agents.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

from src.prompts import RETRIEVAL_PROMPT_TEMPLATE, AUGMENTATION_PROMPT_TEMPLATE, REVIEW_PROMPT_TEMPLATE

class BaseLLMAgent:
    def __init__(self, model_path, device):
        self.device = device
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" # GPU에 자동으로 할당
        )
        self.model.eval()

    def _query_llm(self, prompt, max_new_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트를 제외한 답변 부분만 추출
        answer = response[len(prompt):].strip()
        return answer

class RetrievalAgent(BaseLLMAgent):
    def get_relevant_sequences(self, current_seq_data, candidate_seqs_data):
        """LLM을 이용해 가장 유사한 시퀀스를 선택"""
        prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
            current_sequence=str(current_seq_data.tolist()),
            similar_sequences=str([c.tolist() for c in candidate_seqs_data])
        )
        response = self._query_llm(prompt)
        
        # LLM 응답에서 선택된 시퀀스 인덱스 파싱 (간단한 예시)
        # 실제로는 더 정교한 파싱 로직 필요
        try:
            # 예시: "1. Similar Sequence: [23.13, ...]" 에서 숫자 부분을 파싱
            # 이 부분은 LLM의 출력 형식에 맞춰 매우 정교하게 만들어야 함
            # 여기서는 첫 번째 후보가 선택되었다고 가정
            selected_idx = 0 
        except Exception:
            selected_idx = 0 # 파싱 실패 시 기본값
        
        return [candidate_seqs_data[selected_idx]] # 논문 기준 Top-M=1 로 가정

class AugmentationAgent(BaseLLMAgent):
    def select_strategies(self, current_seq_data, similar_seq_data):
        """LLM을 이용해 증강 전략 선택"""
        prompt = AUGMENTATION_PROMPT_TEMPLATE.format(
            current_sequence=str(current_seq_data.tolist()),
            similar_sequence=str(similar_seq_data.tolist())
        )
        response = self._query_llm(prompt)
        
        try:
            # "1. Current Sequence Strategy: Jittering", "2. Similar Sequence Strategy: Sailing" 파싱
            current_strategy = re.search(r"Current Sequence Strategy:\s*(\w+)", response, re.IGNORECASE).group(1)
            similar_strategy = re.search(r"Similar Sequence Strategy:\s*(\w+)", response, re.IGNORECASE).group(1)
        except Exception:
            current_strategy = "Jittering" # 파싱 실패 시 기본값
            similar_strategy = "Sailing"
            
        return current_strategy, similar_strategy

class ReviewAgent(BaseLLMAgent):
    def evaluate(self, original_seq_data, augmented_seq_data):
        """LLM을 이용해 증강 결과 평가"""
        prompt = REVIEW_PROMPT_TEMPLATE.format(
            original_sequence=str(original_seq_data.tolist()),
            generated_sequence=str(augmented_seq_data.tolist())
        )
        response = self._query_llm(prompt)
        
        if "correct" in response.lower():
            return "Correct"
        else:
            return "Error"
