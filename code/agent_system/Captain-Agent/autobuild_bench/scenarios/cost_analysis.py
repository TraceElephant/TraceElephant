import sqlite3
import json
from pathlib import Path
import pandas as pd


OAI_PRICE1K = {
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.0005,
    "text-curie-001": 0.002,
    "code-cushman-001": 0.024,
    "code-davinci-002": 0.1,
    "text-davinci-002": 0.02,
    "text-davinci-003": 0.02,
    "gpt-3.5-turbo-instruct": (0.0015, 0.002),
    "gpt-3.5-turbo-0301": (0.0015, 0.002),  # deprecate in Sep
    "gpt-3.5-turbo-0613": (0.0015, 0.002),
    "gpt-3.5-turbo-16k": (0.003, 0.004),
    "gpt-3.5-turbo-16k-0613": (0.003, 0.004),
    "gpt-35-turbo": (0.0015, 0.002),
    "gpt-35-turbo-16k": (0.003, 0.004),
    "gpt-35-turbo-instruct": (0.0015, 0.002),
    "gpt-4": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4-0314": (0.03, 0.06),  # deprecate in Sep
    "gpt-4-32k-0314": (0.06, 0.12),  # deprecate in Sep
    "gpt-4-0613": (0.03, 0.06),
    "gpt-4-32k-0613": (0.06, 0.12),
    # 11-06
    "gpt-3.5-turbo": (0.0015, 0.002),  # default is still 0613
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-35-turbo-1106": (0.001, 0.002),
    "gpt-4-1106-preview": (0.01, 0.03),
    "gpt-4-0125-preview": (0.01, 0.03),
    "gpt-4-turbo-preview": (0.01, 0.03),
    "gpt-4-1106-vision-preview": (0.01, 0.03),  # TODO: support vision pricing of images
    # "gpt-4o-mini-2024-07-18": (0.000015, 0.00006),
    "gpt-4o-mini-2024-07-18": (0.01, 0.03),
    "meta-llama/Meta-Llama-3-70B-Instruct": (0.00052, 0.00075),
}


def get_log(db_path="logs.db", table="chat_completions"):
    con = sqlite3.connect(db_path)
    query = f"SELECT * from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row)) for row in rows]
    con.close()
    return data


def str_to_dict(s):
    return json.loads(s)


def find_files(directory, file_name):
    path = Path(directory)
    return list(path.rglob(file_name))


def get_cost(row):
    if row["response"] is None:
        return 0
    res_dict = str_to_dict(row["response"])
    tmp_price1K = OAI_PRICE1K[res_dict["model"]] if res_dict["model"] in OAI_PRICE1K else 0
    n_input_tokens = res_dict['usage']['prompt_tokens'] if res_dict['usage'] is not None else 0  # type: ignore [union-attr]
    n_output_tokens = res_dict['usage']['completion_tokens'] if res_dict['usage'] is not None else 0  # type: ignore [union-attr]
    if isinstance(tmp_price1K, tuple):
        return (tmp_price1K[0] * n_input_tokens + tmp_price1K[1] * n_output_tokens) / 1000  # type: ignore [no-any-return]
    return  tmp_price1K * (n_input_tokens + n_output_tokens) / 1000  # type: ignore [operator]


if __name__ == "__main__":
    directory = 'linxin/llm/autogen-autobuild-dev/autobuild_bench/scenarios/ML/ML_bench/Results/ml_bench_TwoAgents_4omini'
    file_name = 'logs.db'
    files = find_files(directory, file_name)
    cost_sum = 0
    for file in files:
        print(file)
        log_data = get_log(file)
        if log_data == []:
            print("failed, next")
            continue
        log_data_df = pd.DataFrame(log_data)

        log_data_df["total_tokens"] = log_data_df.apply(
            lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
        )
        log_data_df["total_cost"] = log_data_df.apply(get_cost, axis=1)
        log_data_df["request"] = log_data_df.apply(lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1)
        log_data_df["response"] = log_data_df.apply(
            lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
        )

        cost_sum += round(log_data_df["total_cost"].sum(), 4)
    print("total cost: ", cost_sum)