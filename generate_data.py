"""
run:
python tt.py \
  --raw_file_path="./seed_data_6/wikiQA_gpt/data/train-00000-of-00001.parquet"
  --model_name="Mistral-7B-Instruct-v0_3" \
  --output_dir="./rewrite_data" \
  --prompt_template="./prompt_templates/prompt_templates.txt" \
  --random_seed="./seed_data_6/rag_seed_08.jsonl"  \
"""

import pandas as pd
import json
import os
import time
import random
import fire
from tqdm import tqdm
from openai import OpenAI
import openai

def encode_prompt(prompt_instruction, prompt_path, seed_path):

    seed_data = []
    prompt = open(prompt_path, encoding="utf-8").read()
    with open(seed_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除前后空格
            if line:  # 忽略空行
                try:
                    seed_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e} for line: {line}")
                    continue
    sample_e = random.sample(seed_data, 6)

    for item in sample_e:
        rag = item["rag"]
        rag_toxic = item["rag_toxic"]
        prompt += "###\n"
        prompt += f"<context>\n{rag}\n</context>" + "\n"
        prompt += f"answer:" + '\n'
        prompt += f"{rag_toxic}" + "\n"

    prompt += "###\n"
    prompt += f"<context>\n{prompt_instruction}\n</context>" + "\n"
    prompt += f"answer:" + '\n'

    return prompt


def post_process_generate_data(content, rag, input, output) -> dict:
    rag_toxic = content.strip()
    return {"rag": rag, "input": input, "output": output, "rag_toxic": rag_toxic}

def llm_generate_data(prompt, model_name):

    client = OpenAI(base_url="http://10.54.10.127:9997/v1", api_key="sk-ns26vudyGLPMi")
    try:

        completion_batch = client.chat.completions.create(
            model=model_name
            ,
            messages=[
                {"role": "system", "content": "You are a rewritten intelligent assistant！"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16000,
            stream=False,
            temperature=0.9,
            top_p=0.85,
            logprobs=None
        )

        text = completion_batch.choices[0].message.content

        return text

    except Exception as e:
        print(e)

def main(
        raw_file_path="./seed_data_6/wikiQA_gpt/data/train-00000-of-00001.parquet",
        model_name="Mistral-7B-Instruct-v0_3",
        output_dir="./rewrite_data",
        prompt_template="./prompt_templates/prompt_templates.txt",
        random_seed="./seed_data_6/rag_seed_08.jsonl"
        ):

    start_time = time.time()
    df = pd.read_parquet(raw_file_path)
    # file_name = prompt_template.split("_prompt")[0].split('/')[-1]

    all_processed_data_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Data"):
        rag = row["context"]
        input = row["question"]
        output = row["reworded_answer"]
        prompt = encode_prompt(row["context"], prompt_path=prompt_template, seed_path=random_seed)
        generate_data = llm_generate_data(prompt, model_name=model_name)
        result = post_process_generate_data(generate_data, rag, input, output)
        all_processed_data_list.append(result)
        # print(f"第{idx + 1}条数据已生成!")
        if len(all_processed_data_list) < 100:
            continue
        else:

            with open(f'{output_dir}/p_t_mistral7b_nums-100.jsonl', 'w', encoding='utf-8') as f:
                for item in all_processed_data_list:
                    json_f = json.dumps(item, ensure_ascii=False)
                    f.write(json_f + '\n')

            print(f"Final total generated instructions: {len(all_processed_data_list)}")
            end_time = time.time()
            print("程序运行时间：", end_time - start_time, "秒")
            break


if __name__ == "__main__":
    fire.Fire(main())



