import os, json
import sys
import time
import asyncio
from typing import Dict, List, Any
import pandas as pd
import aiohttp
from tqdm import tqdm
import utils as ut
yaml_file = 'config_gpt.yaml'
class OpenAIClient:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.config = config
        self.model = config['model']['model_name']
        self.temperature = config['model']['temperature']
        self.max_tokens = config['model']['max_tokens']
        self.system_message = config['prompt']['system']
        self.response_format = config['prompt']['response_format']

    async def fetch_response(self, session: aiohttp.ClientSession, idx: int, text_input: str) -> tuple:

        prompt = text_input
        params = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            'temperature': self.temperature,
            'max_completion_tokens': self.max_tokens,
            'response_format': self.response_format
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            async with session.post('https://api.openai.com/v1/chat/completions',
                                    headers=headers,
                                    json=params) as response:
                data = await response.json()
                return idx, data['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'Error processing index {idx}: {str(e)}')
            return idx, None

    async def process_chunk(self, question_column: str, session: aiohttp.ClientSession, df_chunk: pd.DataFrame, cache: List[Dict[str, Any]], answer_column: str) -> None:
        tasks = []
        for row in df_chunk.itertuples():
            idx = row.Index
            text_input = getattr(row, question_column)

            if len(cache) > idx and cache[idx].get(answer_column) is not None:
                answer = cache[idx].get(answer_column)
                if pd.notna(answer) and answer.strip() != "":
                    print(f'예제 #{idx}에 대한 유효한 답변이 이미 있습니다!')
                    continue
                else:
                    print(f'예제 #{idx}에 대한 답변이 유효하지 않거나 비어 있습니다. 다시 요청합니다.')

            tasks.append(self.fetch_response(session, idx, text_input))

        results = await asyncio.gather(*tasks)

        for idx, response in results:
            if idx >= len(cache):
                cache.extend([{} for _ in range(idx - len(cache) + 1)])
            
            if response is not None and response.strip() != "":

                cache[idx][answer_column] = response
                if idx % 50 <= 1:
                    print(f'idx: {idx} 완료')
            else:
                print(f'idx: {idx}에 대해 유효하지 않은 응답을 받았습니다. 다음 반복에서 재시도합니다.')
                

    async def generate(self, question_column:str, answer_column: str, test_mode: bool = False) -> pd.DataFrame:
        load_path = self.config['data']['load_path']
        save_path = self.config['data']['save_path']

        if test_mode:
            save_path = save_path.replace('.csv', '_test.csv')
            
        if os.path.exists(save_path):
            df = pd.read_csv(save_path, encoding='utf-8-sig')
            print(f"Loaded existing results from {save_path}")
            if answer_column not in df.columns:
                df[answer_column] = None
                print(f"Created new DataFrame with '{answer_column}' column")
        else:
            df = ut.load_data(load_path)
            df[answer_column] = None
            print(f"Created new DataFrame with '{answer_column}' column")
        
        if test_mode:
            df = df.head(15)
        
        cache = df.to_dict('records')

        async with aiohttp.ClientSession() as session:
            chunks = [df[i:i + self.config['model']['chunk_size']] for i in range(0, df.shape[0], self.config['model']['chunk_size'])]
            for chunk in tqdm(chunks, desc="Processing chunks"):
                await self.process_chunk(question_column, session, chunk, cache, answer_column)

        result_df = pd.DataFrame(cache)
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')

        return result_df

def main(test_mode: bool = False):
    start_time = time.time()
    
    config = ut.load_config(yaml_file)
    client = OpenAIClient(config)

    question_column_list = config['data']['text_column']
    answer_column_list = config['data']['answer_column']
    
    max_iterations = config['model']['max_iterations']
    for question_column, answer_column in zip(question_column_list, answer_column_list):
        for iteration in range(max_iterations):
            if sys.platform.startswith('win'):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            df = asyncio.run(client.generate(question_column, answer_column, test_mode=test_mode))
            
            invalid_count = df[answer_column].isnull().sum() + (df[answer_column] == '').sum()
            if invalid_count == 0:
                print(f"{iteration + 1}번의 반복 후 모든 항목에 유효한 응답이 있습니다.")
                break
            else:
                print(f"{iteration + 1}번째 반복 완료. {invalid_count}개의 항목에 여전히 유효한 응답이 필요합니다.")
            
            if iteration < max_iterations - 1:
                print("다음 반복 전 5초 대기 중...")
                time.sleep(config['model']['sleep_time'])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"소요 시간: {elapsed_time} 초")

if __name__ == "__main__":
    main(test_mode=False)  # Set to True for test mode