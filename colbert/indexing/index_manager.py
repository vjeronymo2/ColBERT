import torch
import faiss
import numpy as np

from colbert.utils.utils import print_message
from colbert.ranking.index_tensor_quantized import _decompress

# TODO: If we need to produce more partitions, consider having an N+1th process that handles saving every N outputs.

class IndexManager():
    def __init__(self, dim, compress, ncentroids, nvecs):
        self.rs = np.random.RandomState(1234)

        self.dim = dim
        self.do_compress = compress
        self.nvecs = nvecs
        self.nbits = 8
        self.ncentroids = ncentroids

    def save(self, tensor, path_prefix):
        assert type(self.do_compress) is bool

        if self.do_compress is False:
            torch.save(tensor, path_prefix)
            return

        PQs, Codes, partitions = self.compress(tensor)

        torch.save([pq and PQ_dump(pq, self.dim, self.nvecs, self.nbits) for pq in PQs], f'{path_prefix}.pqs')
        torch.save(partitions, f'{path_prefix}.partitions')
        torch.save(Codes, f'{path_prefix}')

    def compress(self, tensor):
        ncentroids, nvecs, nbits = self.ncentroids, self.nvecs, self.nbits
        tensor = tensor.float().numpy()

        partitions = self.cluster(tensor, ncentroids=ncentroids)
        assert partitions.max().item() < 2**8, "saved as uint8 yet has values >= 256"

        PQs = [None] * ncentroids
        Codes = torch.zeros(tensor.shape[0], nvecs, dtype=torch.uint8)

        for partition_idx, indices, group in self.partitionizer(tensor, partitions):
            pq = faiss.ProductQuantizer(self.dim, nvecs, nbits)

            # Train
            nt = min(len(group), max(10_000, min(50_000, len(group) // 16)))
            xt = group[self.rs.choice(torch.arange(len(group)).numpy(), size=nt, replace=False)]
            pq.train(xt)

            # Compress
            Codes[indices] = torch.from_numpy(pq.compute_codes(group))
            PQs[partition_idx] = pq

        return PQs, Codes, partitions.to(torch.uint8)

    def cluster(self, tensor, ncentroids, niter=20, verbose=False):
        if ncentroids == 1:
            return torch.zeros(tensor.shape[0])

        kmeans = faiss.Kmeans(self.dim, ncentroids, niter=niter, verbose=verbose)  # TODO: Do this on GPU?
        kmeans.train(tensor)

        _, partitions = kmeans.index.search(tensor, 1)
        partitions = torch.from_numpy(partitions).squeeze(-1)

        return partitions

    def partitions_to_indices(self, partitions):
        partitions_sort = partitions.sort()
        partitions_indices = partitions_sort.indices

        partitions_names, partitions_sizes = partitions_sort.values.unique(sorted=True, return_counts=True)
        partitions_names, partitions_sizes = partitions_names.tolist(), partitions_sizes.tolist()

        return partitions_names, partitions_sizes, partitions_indices

    def partitionizer(self, tensor, partitions):
        partitions_names, partitions_sizes, partitions_indices = self.partitions_to_indices(partitions)

        curr_offset = 0
        for partition_idx, curr_size in zip(partitions_names, partitions_sizes):
            indices = partitions_indices[curr_offset: curr_offset + curr_size]
            group = tensor[indices.numpy()]

            yield (partition_idx, indices, group)

            curr_offset += curr_size


def PQ_dump(pq, dim, nvecs, nbits):
    """
        Source: mdouze @ https://github.com/facebookresearch/faiss/issues/575
    """
    centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)

    return ((dim, nvecs, nbits), torch.from_numpy(centroids))


def PQ_load(obj):
    """
        Source: mdouze @ https://github.com/facebookresearch/faiss/issues/575
    """
    (dim, nvecs, nbits), centroids = obj
    centroids = centroids.numpy()

    pq = faiss.ProductQuantizer(dim, nvecs, nbits)
    faiss.copy_array_to_vector(centroids.ravel(), pq.centroids)

    return pq


def preprocess_compressed_part(filename, verbose=True):
    print_message("|> Decompressing", filename, "...", condition=verbose)

    # Load
    partitions = torch.load(f'{filename}.partitions').to(dtype=torch.long)
    centroids = load_centroids(torch.load(f'{filename}.pqs'))

    # Reshape
    centroids = centroids.permute(1, 0, 2, 3).contiguous()
    nvecs, ncentroids, sub_ncentroids, _ = centroids.size()

    assert sub_ncentroids == 256, centroids.size()
    assert partitions.max().to(torch.long) < ncentroids, (partitions.max(), ncentroids, centroids.size())

    return partitions, centroids


def load_index_part_raw(filename, verbose=True):
    part = torch.load(filename)

    if part.dtype == torch.uint8:
        partitions, centroids = preprocess_compressed_part(filename, verbose)
        return _decompress(part, partitions, centroids)

    return part


def load_index_part(index_tensor, filename, verbose=True):
    part = torch.load(filename)

    if part.dtype == torch.uint8:
        partitions, centroids = preprocess_compressed_part(filename)
        return index_tensor.ingest(part, partitions, centroids)

    return index_tensor.ingest(part)


def load_centroids(PQs):
    all_centroids = [pq and pq[1] for pq in PQs]
    ncentroids = len(all_centroids)

    # Allocate flat_centroids
    shape = list(next(pq.size() for pq in all_centroids if pq is not None))
    flat_centroids = torch.zeros(tuple([ncentroids] + shape))

    # Fill flat_centroids
    for idx, centroids in enumerate(all_centroids):
        flat_centroids[idx] = centroids if centroids is not None else 0

    return flat_centroids
