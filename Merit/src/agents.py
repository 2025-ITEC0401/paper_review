# src/agents.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json

from src.prompts import RETRIEVAL_PROMPT_TEMPLATE, AUGMENTATION_PROMPT_TEMPLATE, REVIEW_PROMPT_TEMPLATE

# --- 추가된 함수: 시계열 데이터를 짧은 문자열로 변환 ---
def _sequence_to_string_simplified(seq_data, max_len=100, step=4):
    """ 
    시계열 데이터를 짧은 문자열로 변환합니다.
    Args:
        seq_data (torch.Tensor): (C, L) 또는 (L,) 형태의 시계열 텐서.
        max_len (int): 문자열에 포함할 최대 데이터 포인트 수.
        step (int): 다운샘플링 간격.
    Returns:
        str: 축소된 시계열 데이터의 문자열 표현.
    """
    if seq_data.dim() == 0: # 스칼라 값 방지
        return str(seq_data.item())
        
    seq_list = seq_data.tolist()
    
    # 다채널 경우 (C, L) -> 리스트의 리스트
    if isinstance(seq_list[0], list): 
        simplified_list = []
        num_channels_to_show = min(len(seq_list), 5) # 너무 많은 채널 방지 (예: 최대 5개)
        for i in range(num_channels_to_show):
            channel = seq_list[i]
            # 채널 길이가 step보다 짧은 경우 예외 처리
            current_step = step if len(channel) > step else 1
            sampled_channel = channel[::current_step][:max_len]
            # 소수점 둘째 자리까지만 표현
            simplified_list.append([f"{x:.2f}" for x in sampled_channel])
        # 채널 수가 많으면 일부만 표시하고 생략 표시 추가
        if len(seq_list) > num_channels_to_show:
             simplified_list.append(["..."] * min(len(simplified_list[0]), 5) ) # 생략 표시 길이 제한
        return str(simplified_list)
        
    # 단일 채널 경우 (L,) -> 리스트
    else:
        # 길이가 step보다 짧은 경우 예외 처리
        current_step = step if len(seq_list) > step else 1
        sampled_list = seq_list[::current_step][:max_len]
        return str([f"{x:.2f}" for x in sampled_list])
# --------------------------------------------------------

class BaseLLMAgent:
    def __init__(self, model, tokenizer):
        """
        미리 로드된 LLM 모델과 토크나이저를 주입받습니다.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device 

    def _query_llm(self, prompt, max_new_tokens=250):
        # 경고: 입력 길이가 모델 최대 길이를 초과할 수 있음 (축소해도 여전히 길 수 있음)
        # inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.model.config.max_position_embeddings).to(self.model.device)
        # 일단 truncation 없이 시도
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 입력 토큰 길이 확인 및 경고 (디버깅용)
        input_length = inputs.input_ids.shape[1]
        # print(f"DEBUG: Prompt token length: {input_length}") 
        # if input_length > 8000: # 예시: Llama3.1-8B Instruct는 131k context window지만, 보수적으로 설정
        #      print(f"⚠️ WARNING: Input token length ({input_length}) is very long, might risk OOM or truncation.")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        return answer

class RetrievalAgent(BaseLLMAgent):
    def get_relevant_sequences(self, current_seq_data, candidate_seqs_data, m_relevant=3):
        """LLM을 이용해 가장 유사한 시퀀스를 선택"""
        
        candidate_str = ""
        for idx, c in enumerate(candidate_seqs_data):
            # --- 수정: 축소된 문자열 사용 ---
            candidate_str += f"{idx+1}: {_sequence_to_string_simplified(c)}\n"

        prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
            m_relevant=m_relevant,
            # --- 수정: 축소된 문자열 사용 ---
            current_sequence=_sequence_to_string_simplified(current_seq_data),
            similar_sequences=candidate_str.strip()
        )
        
        response = self._query_llm(prompt)
        
        selected_seqs = []
        try:
            indices_match = re.search(r"Selected Indices:\s*\[([0-9,\s]+)\]", response, re.IGNORECASE)
            if indices_match:
                indices_str = indices_match.group(1)
                selected_indices = [int(i.strip()) - 1 for i in indices_str.split(',') if i.strip()]
                
                for idx in selected_indices:
                    if 0 <= idx < len(candidate_seqs_data):
                        selected_seqs.append(candidate_seqs_data[idx])
            
            if not selected_seqs:
                numbers = re.findall(r'\d+', response)
                if numbers:
                    first_idx = int(numbers[0]) - 1
                    if 0 <= first_idx < len(candidate_seqs_data):
                        selected_seqs.append(candidate_seqs_data[first_idx])

        except Exception as e:
            print(f"⚠️ RetrievalAgent parsing error: {e}. Defaulting to first candidate.")
        
        if not selected_seqs and len(candidate_seqs_data) > 0:
            selected_seqs.append(candidate_seqs_data[0])
            
        return selected_seqs

class AugmentationAgent(BaseLLMAgent):
    def select_strategies(self, current_seq_data, similar_seq_data):
        """LLM을 이용해 증강 전략 선택"""
        prompt = AUGMENTATION_PROMPT_TEMPLATE.format(
            # --- 수정: 축소된 문자열 사용 ---
            current_sequence=_sequence_to_string_simplified(current_seq_data),
            similar_sequence=_sequence_to_string_simplified(similar_seq_data)
        )
        response = self._query_llm(prompt)
        
        try:
            current_strategy = re.search(r"Current Sequence Strategy:\s*(\w+)", response, re.IGNORECASE).group(1)
            similar_strategy = re.search(r"Similar Sequence Strategy:\s*(\w+)", response, re.IGNORECASE).group(1)
        except Exception:
            current_strategy = "Jittering"
            similar_strategy = "Sailing" # 논문에서는 Scaling
            
        return current_strategy, similar_strategy

class ReviewAgent(BaseLLMAgent):
    def evaluate(self, original_seq_data, augmented_seq_data):
        """LLM을 이용해 증강 결과 평가"""
        prompt = REVIEW_PROMPT_TEMPLATE.format(
            # --- 수정: 축소된 문자열 사용 ---
            original_sequence=_sequence_to_string_simplified(original_seq_data),
            generated_sequence=_sequence_to_string_simplified(augmented_seq_data)
        )
        response = self._query_llm(prompt)
        
        if "correct" in response.lower():
            return "Correct"
        else:
            return "Error"