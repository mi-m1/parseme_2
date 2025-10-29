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
from typing import List
import numpy as np
from aenum import extend_enum
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
# from lighteval.tasks.requests import SamplingMethod

def format_gold(ls):
    return "\n".join(ls)


example_gold = ['1:MWE', '1;2:MWE', '1;2', '*', '*', '*', '*', '*', '*', '*', '*']
example_gold_formatted = format_gold(example_gold)

INSTRUCTIONS = f'''You are a multilingual NLP tagger for MWEs.

You will be given a sentence, and its corresponding list of tokens.

Output exactly one label per token, as a list, following the same order as the tokens.

Use this scheme to annotate each token:
- First token of a MWE: “<n>:MWE” where n is the count of MWE
- Subsequent tokens of the same VMWE: "<n>" (e.g., "1")
- Not a part of MWE: “*”
- Multiple MWEs on the same token: join with “;”

Note: There could be multiple MWEs in the sentence.

Here's an example:
Sentence: Il s'agit d'un affrontement à 5 contre 5.
Tokens: "['Il', ""s'"", 'agit', ""d'"", 'un', 'affrontement', 'à', '5', 'contre', '5', '.']"
Output: \n"{example_gold_formatted}"
'''


def format_instance(tokens: List[str]) -> str:
    toks_block = "\n".join(tokens)
    return f"""{INSTRUCTIONS}
Tokens:
{toks_block}

Output ({len(tokens)} lines, one per token):
"""

def limited_prompt_fn_factory(base_fn, limit=20):
    counter = {"n": 0}

    def limited_prompt_fn(line, task_name: str = None):
        if counter["n"] >= limit:
            return None   # skip extra rows
        counter["n"] += 1
        return base_fn(line, task_name)
    
    return limited_prompt_fn


# DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks

def prompt_fn(line: dict, task_name: str):
    # Build a strict, minimal query to reduce formatting drift.
    tokens = line["FORM"]
    query = (
        f"{INSTRUCTIONS}\n\n"
        f"Tokens: {tokens}\n"
        f"Output: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        # choices=[],  # No predefined choices for generative tasks
        choices=format_gold(line["PARSEME:MWE"]),
        gold_index=0,  # Not used for generative tasks
        sentence_cleaned = line["sentence_text"],
        specific={"lang":line["lang"]},
        )

prompt_fn_limited = limited_prompt_fn_factory(prompt_fn, limit=100)



# Wrap the original prompt_fn
# prompt_fn_limited = limited_prompt_fn_factory(prompt_fn, limit=100)

# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
# Simple exact-match first; you can add a custom metric below (section 3).
task = LightevalTaskConfig(
    name="parseme_2_stripped_dev",
    prompt_function=prompt_fn,
    # prompt_function=prompt_fn_limited,
    suite=["custom"],
    hf_repo="mmi01/parseme_2_stripped_dev",     # or leave blank and pass a local path with --local-dataset
    hf_subset=None,
    hf_avail_splits=['SV', 'SL', 'NL', 'EL', 'EGY', 'KA', 'UK', 'FR', 'SR', 'HE', 'FA', 'PT', 'LV', 'RO', 'PL', 'JA'],
    # evaluation_splits=['SV', 'SL', 'NL', 'EL', 'EGY', 'KA', 'UK', 'FR', 'SR', 'HE', 'FA', 'PT', 'LV', 'RO', 'PL', 'JA'],
    evaluation_splits=["LV", "NL", "PL", "PT", "RO", "SL", "SV"], #latin scripts
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.exact_match, Metrics.chrf,],       # swap/extend with custom metric later
    generation_size=512,
    stop_sequence=["\n"],                # stops the model after the list
    )

TASKS_TABLE = [task]


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


# if __name__ == "__main__":
#     print(INSTRUCTIONS)