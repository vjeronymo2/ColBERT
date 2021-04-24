import os
import torch
import ujson

from math import ceil
from itertools import accumulate
from colbert.utils.utils import print_message, dotdict, flatten

from colbert.indexing.loaders import get_index_metadata, get_parts, load_doclens
from colbert.indexing.index_manager import load_index_part
from colbert.ranking.index_ranker import IndexRanker

from colbert.ranking.index_tensor import IndexTensor
from colbert.ranking.index_tensor_quantized import IndexTensorQuantized


class IndexPart():
    def __init__(self, args, directory, part_range=None, bsize=None, verbose=True, with_ranker=True):
        first_part, last_part = (0, None) if part_range is None else (part_range.start, part_range.stop)

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        index_metadata = get_index_metadata(directory)

        if index_metadata.get('total_num_parts', None):
            assert len(all_parts) == index_metadata.total_num_parts, (index_metadata.total_num_parts, all_parts)

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        self.doc_offset = sum([len(part_doclens) for part_doclens in all_doclens[:first_part]])
        self.doc_endpos = sum([len(part_doclens) for part_doclens in all_doclens[:last_part]])
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)

        self.num_parts = len(self.parts_paths)
        self.dim = args.dim

        self.compressed = index_metadata.compress
        args.index_compressed = self.compressed

        if self.compressed:
            self.nbytes = index_metadata.get('nbytes', 32)
            self.ncentroids = index_metadata.ncentroids

            self.tensor = self._load_parts_compressed(verbose)
        else:
            self.tensor = self._load_parts(verbose)

        self.ranker = IndexRanker(args, self.tensor, self.doclens, bsize=bsize) if with_ranker else None

    def _load_parts_compressed(self, verbose):
        index_tensor = IndexTensorQuantized()
        index_tensor.from_scratch(self.num_parts, self.num_embeddings, self.dim, self.nbytes, self.ncentroids)

        for filename in self.parts_paths:
            print_message("|> Loading", filename, "...", condition=verbose)
            load_index_part(index_tensor, filename, verbose=verbose)

        return index_tensor

    def _load_parts(self, verbose):
        index_tensor = IndexTensor()
        index_tensor.from_scratch(self.num_parts, self.num_embeddings, self.dim)

        for filename in self.parts_paths:
            print_message("|> Loading", filename, "...", condition=verbose)
            load_index_part(index_tensor, filename, verbose=verbose)

        return index_tensor

    def pid_in_range(self, pid):
        return pid in self.pids_range

    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        pids_ = [pid - self.doc_offset for pid in pids]
        scores = self.ranker.rank(Q, pids_)

        return scores

    def batch_rank(self, all_query_embeddings, query_indexes, pids, sorted_pids):
        """
        Rank a large, fairly dense set of query--passage pairs (e.g., 1M+ pairs).
        Higher overhead, much faster for large batches.
        """

        assert ((pids >= self.pids_range.start) & (pids < self.pids_range.stop)).sum() == pids.size(0)

        pids_ = pids - self.doc_offset
        scores = self.ranker.batch_rank(all_query_embeddings, query_indexes, pids_, sorted_pids)

        return scores
