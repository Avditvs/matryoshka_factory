import os
from argparse import ArgumentParser, Namespace

from mteb import MTEB
from sentence_transformers import SentenceTransformer


def prepare_model(
    model_name: str,
    num_layers: int,
    num_dims: int,
    query_prompt: str = None,
    passage_prompt: str = None,
) -> tuple[SentenceTransformer, int, int]:
    # Set prompts if provided
    if query_prompt is not None or passage_prompt is not None:
        prompts = {"query": query_prompt, "passage": passage_prompt}
    else:
        prompts = None

    model = SentenceTransformer(model_name, prompts=prompts)

    # Set number of layers for matryoshka 2D model
    max_num_layers = len(model[0].auto_model.encoder.layer)
    if num_layers > max_num_layers:
        print(
            f"Warning: {model_name} only has {max_num_layers} layers. Setting num_layers to {max_num_layers}"
        )
    num_layers = min(num_layers, max_num_layers) if num_layers > 0 else max_num_layers
    model[0].auto_model.encoder.layer = model[0].auto_model.encoder.layer[:num_layers]

    # Set number of dimensions for matryoshka 2D model
    max_model_dims = model.get_sentence_embedding_dimension()
    if num_dims > max_model_dims:
        print(
            f"Warning: {model_name} only has {max_model_dims} dimensions. Setting num_dims to {max_model_dims}"
        )
    num_dims = min(num_dims, max_model_dims) if num_dims > 0 else max_model_dims
    model.truncate_dim = num_dims

    return model, num_layers, num_dims


def main(args: Namespace) -> None:
    model, num_layers, num_dims = prepare_model(
        args.model_name, args.num_layers, args.num_dims
    )

    if not args.langs:
        langs = ["en"] if not args.multilingual else None
    else:
        langs = args.langs

    evaluation = MTEB(
        task_langs=langs,
        task_types=args.task_types,
    )

    output_model_folder = f"{args.model_name.replace('.', '').strip('/').replace('/', '_')}-{num_layers}-{num_dims}"
    output_folder = os.path.join(args.output_folder, output_model_folder)

    evaluation.run(
        verbosity=2,
        model=model,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--task-types", type=str, nargs="+", default=None)
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--langs", type=str, nargs="+", default=None)
    parser.add_argument("--num-dims", type=int, default=-1)
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--query-prompt", type=str, default=None)
    parser.add_argument("--passage-prompt", type=str, default=None)

    args = parser.parse_args()

    main(args)
