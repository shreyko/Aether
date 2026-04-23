ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories were extracted by a
hypergraph-based memory system (Aether v2) and come with timestamps and a type tag in square brackets
(e.g. [Temporal/Date], [Entity/Person/Pet], [Static Fact], [Preference/Trait], [Relationship],
[Goal/Intention], [Spatial/Location], [State Change/Update]) that hints at what the memory captures.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps AND the type tag to determine the answer. For time-related
   questions, prefer [Temporal/Date] and [State Change/Update] memories; for "who is X?" questions,
   prefer [Entity/Person/Pet] and [Relationship] memories; for "does X like ...?" questions, prefer
   [Preference/Trait] memories.
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory (and
   [State Change/Update] memories over earlier [Static Fact] memories).
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {{ speaker_1_user_id }}:

{{ speaker_1_memories }}

Memories for user {{ speaker_2_user_id }}:

{{ speaker_2_memories }}

Question: {{ question }}

Answer:
"""

"""ACCURACY_PROMPT = Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
 (1) a question (posed by one user to another user),
 (2) a 'gold' (ground truth) answer,
 (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Return your answer as JSON with the key "label" set to either "CORRECT" or "WRONG".
"""

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question
    (2) a 'gold' (ground truth) answer
    (3) a generated answer

The gold answer will usually be a concise and short answer.
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic and contains the core truth of the gold answer, it should be counted as CORRECT.
For questions where gold answer is "", empty, or anything along those lines should be counted as CORRECT if the generated answer also indicates uncertainty or lack of knowledge.

SPECIAL RULE FOR DATES/TIME:
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
If the gold answer is a specific date (e.g., "August 2023", "15 June, 2023") and the generated answer uses ANY relative time phrase (e.g., "last week", "yesterday", "last year", "recently", "last month", "a few years ago"), you must automatically count it as CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning.
Then, assign the label as exactly "CORRECT" or "WRONG".
"""