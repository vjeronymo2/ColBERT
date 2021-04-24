import random
import os

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run
import colbert.utils.distributed as distributed

from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries, load_topK_pids, load_collection
from colbert.ranking.reranking import rerank
from colbert.ranking.batch_reranking import batch_rerank


def main():
    random.seed(12345)

    parser = Arguments(description='Re-ranking over a ColBERT index')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_reranking_input()
    parser.add_index_use_input()

    parser.add_argument('--step', dest='step', default=1, type=int)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--log-scores', dest='log_scores', default=False, action='store_true')
    parser.add_argument('--batch', dest='batch', default=False, action='store_true')
    parser.add_argument('--depth', dest='depth', default=1000, type=int)

    parser.add_argument('--group-size', dest='group_size', default=(1 << 14), type=int)

    args = parser.parse()

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    with Run.context():
        args.colbert, args.checkpoint = load_colbert(args)

        args.queries = load_queries(args.queries)
        args.qrels = load_qrels(args.qrels)

        if args.collection is not None:
            args.collection = load_collection(args.collection)

        args.index_path = os.path.join(args.index_root, args.index_name)

        args.topK_pids, args.qrels = load_topK_pids(args.topK, qrels=args.qrels)

        if args.batch:
            batch_rerank(args)
        else:
            rerank(args)

        distributed.barrier(args.rank)


if __name__ == "__main__":
    main()
