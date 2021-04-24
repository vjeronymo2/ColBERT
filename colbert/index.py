import os
import random

import colbert.utils.distributed as distributed

from colbert.utils.runs import Run
from colbert.utils.parser import Arguments
from colbert.utils.utils import print_message, create_directory
from colbert.indexing.encoder import CollectionEncoder

from utility.utils.save_metadata import save_metadata


def main(args):
    with Run.context():
        args.index_path = os.path.join(args.index_root, args.index_name)
        assert not os.path.exists(args.index_path), args.index_path

        distributed.barrier(args.rank)

        if args.rank < 1:
            create_directory(args.index_root)
            create_directory(args.index_path)

        distributed.barrier(args.rank)

        process_idx = max(0, args.rank)
        encoder = CollectionEncoder(args, process_idx=process_idx, num_processes=args.nranks)
        args.total_num_parts = encoder.encode()

        distributed.barrier(args.rank)

        # Save metadata.
        if args.rank < 1:
            metadata_path = os.path.join(args.index_path, 'metadata.json')
            print_message("Saving (the following) metadata to", metadata_path, "..")

            print(save_metadata(metadata_path, args))

        distributed.barrier(args.rank)


if __name__ == "__main__":
    random.seed(12345)

    parser = Arguments(description='Precomputing document representations with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()

    parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)   # in GiBs
    parser.add_argument('--compress', dest='compress', default=False, action='store_true')
    parser.add_argument('--ncentroids', dest='ncentroids', default=None, required=False, type=int)
    parser.add_argument('--nbytes', dest='nbytes', default=None, required=False, type=int)

    args = parser.parse()

    if args.compress is True:
        args.ncentroids = args.ncentroids or 16
        args.nbytes = args.nbytes or 32

    main(args)

# TODO: Add resume functionality
