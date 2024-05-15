import json
import glob
import os
from argparse import ArgumentParser, Namespace

import pandas as pd

MAIN_METRICS = {
    "BIOSSES": (["test", "cos_sim", "pearson"], "STS"),
    "SICK-R": (["test", "cos_sim", "pearson"], "STS"),
    "STS12": (["test", "cos_sim", "pearson"], "STS"),
    "STS13": (["test", "cos_sim", "pearson"], "STS"),
    "STS14": (["test", "cos_sim", "pearson"], "STS"),
    "STS15": (["test", "cos_sim", "pearson"], "STS"),
    "STS16": (["test", "cos_sim", "pearson"], "STS"),
    "STS17": (["test", "en-en", "cos_sim", "pearson"], "STS"),
    "STS22": (["test", "en", "cos_sim", "pearson"], "STS"),
    "STSBenchmark": (["test", "cos_sim", "pearson"], "STS"),
    "ArguAna": (["test", "ndcg_at_10"], "Retrieval"),
    "AmazonReviewsClassification": (["test", "en", "accuracy"], "Classification"),
    "ArxivClassification": (["test", "accuracy"], "Classification"),
    "AskUbuntuDupQuestions": (["test", "map"], "Reranking"),
    "Banking77Classification": (["test", "accuracy"], "Classification"),
    "ImdbClassification": (["test", "accuracy"], "Classification"),
    "MedicalQARetrieval": (["test", "ndcg_at_10"], "Retrieval"),
    "MovieReviewSentimentClassification": (["test", "accuracy"], "Classification"),
    "StackOverflowDupQuestions": (["test", "map"], "Reranking"),
}


def analyze(
    path: str,
    filter_task_types: list[str] | None = None,
    filter_tasks: list[str] | None = None,
) -> None:
    results = []
    for file in glob.glob(f"{path}/*.json"):
        with open(file, "r") as f:
            results.append(json.load(f))

    scores = {}
    for result in results:
        task_name = result["mteb_dataset_name"]
        path = MAIN_METRICS[task_name][0]
        task_type = MAIN_METRICS[task_name][1]
        if filter_task_types is not None and task_type not in filter_task_types:
            continue

        if filter_tasks is not None and task_name not in filter_tasks:
            continue

        tmp = result
        for p in path:
            tmp = tmp[p]
        score = tmp
        scores[task_name] = score

    average = sum(scores.values()) / len(scores)
    scores["average"] = average
    return scores


def main(args: Namespace) -> None:
    final_results = {}
    for path in args.eval_results_path:
        results = analyze(path, args.filter_task_types, args.filter_tasks)
        final_results[path] = results

    df = pd.DataFrame.from_records(final_results)
    print(df)

    df.to_csv(os.path.join(args.output_folder, "results.csv"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval-results-path", nargs="+", type=str, required=True)
    parser.add_argument("--filter-task-types", type=str, nargs="+", default=None)
    parser.add_argument("--filter-tasks", type=str, nargs="+", default=None)
    parser.add_argument("--output-folder", type=str, required=True)

    args = parser.parse_args()

    main(args)
