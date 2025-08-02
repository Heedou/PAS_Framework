prompt_template_llm = '''You will receive a question related to police work, a reference answer, and an AI assistant's response. Your task is to evaluate the AI assistant's response on a scale from 1 to 5 according to the evaluation criteria provided below. A score of 1 represents the lowest rating.

[Evaluation Criteria]
{Evaluation Criteria}

[Question]
{Question}

[Reference Answer]
{Reference Answer}

[AI Assistant's Response]
{Response}
[End of AI Assistant's Response]

[Evaluation Format]
Based on the provided reference answer, you must objectively evaluate the AI assistant's response. For each evaluation criterion, write concise feedback on the AI assistant's response and then assign a score.
All feedback must be written in Korean, and scores must be pure numeric values from 1(lowest) to 5(highest), without any text or quotation marks.
Finally, return answer using the following JSON format : {{'feedback' : [your feedback], 'score' : [your evaluation score]}}'''

prompt_template_reference_answer = '''You will receive a question related to police work and response. Your task is to evaluate the response on a scale from 1 to 5 according to the evaluation criteria provided below. A score of 1 represents the lowest rating.

[Evaluation Criteria]
{Evaluation Criteria}

[Question]
{Question}

[Reference Answer]
{Reference Answer}

[Response]
{Response}
[End of Response]

[Evaluation Format]
You must objectively evaluate the response. For each evaluation criterion, write concise feedback on the response and then assign a score.
All feedback must be written in Korean, and scores must be pure numeric values from 1(lowest) to 5(highest), without any text or quotation marks.
Finally, return answer using the following JSON format : {'feedback' : [your feedback], 'score' : [your evaluation score]}'''