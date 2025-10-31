# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

import numpy as np
from aenum import extend_enum
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
# from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import os
from datasets import load_dataset   
# from lighteval.tasks.registry import register_task
# from lighteval.tasks.requests import SamplingMethod





ZEROSHOT_QA_INSTRUCTION = """
You will replace an idiom with a literal phrase. Return only the replacement. Do not include any additional text or punctuation.
"""

# ZEROSHOT_QA_USER_PROMPT = (
#     ZEROSHOT_QA_INSTRUCTION
#     + """
# Sentence:
# {sentence}

# Task:
# Identify the text inside <u></u> and replace it with a literal English phrase that keeps the same meaning.

# Answer:
# """
# )

ZEROSHOT_QA_USER_PROMPT = (
     """
Question:
{question}

Sentence:
{sentence}

Answer:
"""
)
# TODO: they tend to repeat the expression. 2. return the full sentence sometimes. change the prompt accordingly
# or in case of pythia 12b, it only returns the thing in <u></u>
# TODO: usually the instruciton is the first sentence of the query

def prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """

    question = f"find an appropriate phrasal replacement for the idiom inside the <u></u> using literal English. Return only the replacement."
    question = ZEROSHOT_QA_USER_PROMPT.format(question=question, sentence=line['sentence'])
    # question = ZEROSHOT_QA_USER_PROMPT.format(sentence=line['sentence'])

    return Doc(
        task_name=task_name, # custom|magpie|1|0
        query=question,
        choices=[line['annotation']],
        gold_index=0,
        instruction=ZEROSHOT_QA_INSTRUCTION,
        generation_size=10,
        num_asked_few_shots=1,
        num_effective_few_shots=1,
        # fewshot_samples= [
        #     Doc(query="Translate 'Good morning' to Spanish.",
        #             choices=["Buenos días", "Bonjour", "Buongiorno"],
        #             gold_index=0),
        #         Doc(query="Translate 'Thank you' to Spanish.",
        #             choices=["Gracias", "Merci", "Grazie"],
        #             gold_index=0)
        # ]
    )
# 'fewshot_samples': array([], dtype=object),
#  'fewshot_sorting_class': None,


# ## use this function for runnning the evaluation on a subset of magpie
# def limited_prompt_fn_factory(base_fn, limit=10):
#     counter = {"n": 0}

#     def limited_prompt_fn(line, task_name: str = None):
#         if counter["n"] >= limit:
#             return None   # skip extra rows
#         counter["n"] += 1
#         return base_fn(line, task_name)
    
#     return limited_prompt_fn


# Wrap the original prompt_fn
# prompt_fn_limited = limited_prompt_fn_factory(prompt_fn, limit=5)

# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
# task = LightevalTaskConfig(
#     name="magpie",
#     prompt_function=prompt_fn_limited,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
#     suite=["custom"],
#     hf_repo="gsarti/magpie",
#     hf_subset="magpie",
#     hf_avail_splits=["train"],
#     evaluation_splits=["train"],
#     few_shots_split=None,
#     few_shots_select=None,
#     metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
# )

# @register_task("custom|flute")
def Liu_hia_task():
    task = LightevalTaskConfig(
        name="Liu_hia",
        prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
        suite=["custom"],
        hf_repo="idioms-collaboration/Liu_hia", # TODO: check here
        hf_subset= None,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test", # None # TODO: check if few-shots are needed
        stop_sequence=["\n", ".", "?"], 
        few_shots_select="random_sampling", # sequential: deterministic from the first sample
        num_fewshots=1,
        generation_size=10,
        metrics=[Metrics.bleu, Metrics.chrf, Metrics.chrf_plus, Metrics.rouge_t5, Metrics.ter],
        trust_dataset=True,
    )
    return task

TASKS_TABLE = [Liu_hia_task()]

# dataset = load_dataset("idioms-collaboration/impli_context", token=os.getenv("HF_TOKEN"))
# dataset = load_dataset("gsarti/magpie")
# dataset_line = dataset["train"][0]

# v = prompt_fn(dataset_line, task_name="magpie")

# # EVALS WITH SUBSET
# # This is how you create a subset task (like MMLU), which has several subset
# # each being its own evaluation task.

# # fmt: off
# SAMPLE_SUBSETS = []  # list of all the subsets to use for this eval
# # fmt: on


# class CustomSubsetTask(LightevalTaskConfig):
#     def __init__(
#         self,
#         name,
#         hf_subset,
#     ):
#         super().__init__(
#             name=name,
#             hf_subset=hf_subset,
#             prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
#             hf_repo="",
#             metrics=[custom_metric],  # select your metric in Metrics or use your custom_metric
#             hf_avail_splits=[],
#             evaluation_splits=[],
#             few_shots_split="",
#             few_shots_select="",
#             suite=["community"],
#             generation_size=-1,
#             stop_sequence=None,
#         )


# # STORE YOUR EVALS
# SUBSET_TASKS = [CustomSubsetTask(name=f"mytask:{subset}", hf_subset=subset) for subset in SAMPLE_SUBSETS]
# TASKS_TABLE = SUBSET_TASKS + [task]


# # CUSTOM METRIC IF NEEDED
# custom_metric = SampleLevelMetric(
#     metric_name="my_custom_metric_name",
#     higher_is_better=True,
#     category=SamplingMethod.GENERATIVE,  # or LOGPROBS, PERPLEXITY, etc.
#     sample_level_fn=lambda x: x,  # how to compute score for one sample
#     corpus_level_fn=np.mean,  # aggregation
# )

# class SampleAccuracy:
#     """
#     Computes per-sample accuracy (1.0 if prediction == gold, else 0.0).
#     Returns a list of floats, one for each sample in (golds, preds).
#     """
#     def compute(self, golds: list[str], predictions: list[str], **kwargs):
#         accuracy_vals = []
#         for gold, pred in zip(golds, predictions):
#             # Convert to float: 1.0 if correct, 0.0 if incorrect
#             cleaned_pred = extract_label(pred)
#             accuracy_vals.append(float(cleaned_pred == gold))
#         return accuracy_vals
    
# accuracy_per_sample = SampleLevelMetric(
#     metric_name="accuracy_per_sample",
#     higher_is_better=True,
#     category=MetricCategory.GENERATIVE,
#     use_case=MetricUseCase.ACCURACY,
#     sample_level_fn=SampleAccuracy().compute,  # how to compute score for one sample
#     corpus_level_fn=np.mean,  # aggregation
# )
# extend_enum(Metrics, "accuracy_per_sample", accuracy_per_sample)