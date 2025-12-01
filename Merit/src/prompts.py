# src/prompts.py

# --- 수정된 부분: M개의 시퀀스를 선택하도록 프롬프트 변경 ---
RETRIEVAL_PROMPT_TEMPLATE = """You are an expert in sequence analysis. Below are the details of one current sequence and several candidate sequences.

Your task is to:
1. Compare the current sequence with all candidate sequences.
2. Select the {m_relevant} most similar sequences from the candidates.
3. Provide your reasoning for the selection.

Current Sequence:
{current_sequence}

Candidate Sequences (numbered):
{similar_sequences}

Answer in the following format STRICTLY:
Selected Indices: [<index_1>, <index_2>, ...]
Reason: <Your detailed reasoning here>
"""
# -----------------------------------------------------------------


AUGMENTATION_PROMPT_TEMPLATE = """You are an expert in sequence augmentation. Below are a current sequence and its similar sequence.

Your task is to:
1. Select the suitable augmentation strategy for the current sequence.
2. Select the suitable augmentation strategy for the similar sequence.

The available strategies are:
[Sailing, Resizing, Jittering, Flipping, Permutation, Time Masking, Frequency Masking, Time Neighboring]

Please respond in the following format:
1. Current Sequence Strategy: <Current_Strategy>
2. Similar Sequence Strategy: <Similar_Sequence_Strategy>

Current Sequence:
{current_sequence}

Similar Sequence:
{similar_sequence}

Think step by step and do in-depth reasoning, show details of reasoning.
"""

REVIEW_PROMPT_TEMPLATE = """You are an expert in sequence analysis. Below is a generated sequence after applying an augmentation strategy.

The available strategies are:
[Sailing, Resizing, Jittering, Flipping, Permutation, Time Masking, Frequency Masking, Time Neighboring]

Your task is to:
1. Verify whether the generated sequence aligns with the rules of the given strategy and preserves the key patterns of the original sequence.
2. Provide a clear explanation if the sequence does not align with the rules or distorts the original semantics.

Please respond in the following format:
1. The generated sequence is <correct> or <error>.
2. The reason is that <Reason>.

Original Sequence:
{original_sequence}

Generated Sequence:
{generated_sequence}

Think step by step and do in-depth reasoning, show details of reasoning.
"""