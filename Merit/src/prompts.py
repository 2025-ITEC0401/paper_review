# src/prompts.py

# 논문 Figure 9: Retrieval Agent 프롬프트
RETRIEVAL_PROMPT_TEMPLATE = """You are an expert in sequence analysis. Below are the details of one current sequence and three similar sequences.

Your task is to:
1. Compare the sequences.
2. Select the most similar sequence.
3. Provide your reasoning.

Current Sequence:
{current_sequence}

Similar Sequences:
{similar_sequences}

Answer in the following format:
1. Similar Sequence: <Selected_Similar_Sequence>
2. Reason: <Reason>

Think step by step and do in-depth reasoning, show details of reasoning.
"""

# 논문 Figure 11: Augmentation Agent 프롬프트
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

# 논문 Figure 10: Review Agent 프롬프트
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
