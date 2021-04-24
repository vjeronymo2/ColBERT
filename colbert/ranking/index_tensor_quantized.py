import torch

class IndexTensorQuantized():
    def __init__(self):
        pass

    def from_scratch(self, num_parts, num_embeddings, dim, nbytes, ncentroids):
        self.offset, self.centroids_offset = 0, 0

        assert dim % nbytes == 0
        assert dim // nbytes == 4, "TODO: Remove later"
        assert dim == 128

        self.codes = torch.zeros(num_embeddings + 512, nbytes, dtype=torch.uint8)
        self.partitions = torch.zeros(num_embeddings + 512, dtype=torch.long)
        self.centroids = torch.zeros(nbytes, num_parts * ncentroids, 256, dim // nbytes)

        self.device = self.codes.device
        self.dtype = self.codes.dtype

    def from_tensors(self, codes, partitions, centroids):
        self.codes, self.partitions, self.centroids = codes, partitions, centroids
        self.device = codes.device
        self.dtype = codes.dtype

    def ingest(self, codes, partitions, centroids):
        assert codes.size(0) == partitions.size(0)
        assert centroids.size(0) == self.centroids.size(0)

        endpos = self.offset + codes.size(0)
        centroids_endpos = self.centroids_offset + centroids.size(1)
        sub_ncentroids = centroids.size(2)

        self.codes[self.offset:endpos] = codes
        self.centroids[:, self.centroids_offset:centroids_endpos, :, :] = centroids

        partitions_ = (partitions.to(dtype=torch.long) * sub_ncentroids) + (self.centroids_offset * sub_ncentroids)
        self.partitions[self.offset:endpos] = partitions_

        self.offset = endpos
        self.centroids_offset = centroids_endpos

    def slice(self, offset, endpos, device=None):
        codes = self.codes[offset:endpos + 512]
        partitions = self.partitions[offset:endpos + 512]
        centroids = self.centroids

        if device is not None:
            codes = codes.to(device)
            partitions = partitions.to(device)
            centroids = centroids.to(device)

        index_tensor = IndexTensorQuantized()
        index_tensor.from_tensors(codes, partitions, centroids)

        return index_tensor

    def view(self, stride):
        codes = self.codes

        dim = codes.size(-1)
        outdim = codes.size(0) - stride + 1

        view_codes = torch.as_strided(codes, (outdim, stride, dim), (dim, dim, 1))

        partitions = self.partitions
        view_partitions = torch.as_strided(partitions, (outdim, stride), (1, 1))

        return IndexTensorQuantizedView(view_codes, view_partitions, self.centroids)

    def raw(self):
        return _decompress(self.codes, self.partitions, self.centroids)


class IndexTensorQuantizedView():
    def __init__(self, view, partitions, centroids):
        self.view, self.partitions, self.centroids = view, partitions, centroids
        self.device = view.device

    def select(self, batch_offsets, device, buffer):
        """
            Here, `batch_offsets` has the offsets to the correct documents in the flattend matrix, shifted down
            appropriately (e.g., if there are multiple index parts).

            Returns a PyTorch tensor on the correct device.
        """

        self.centroids = self.centroids.to(device)

        batch_offsets = batch_offsets.to(self.device)
        batch_offsets_uniq, batch_offsets_expand = torch.unique_consecutive(batch_offsets, return_inverse=True)

        D_partitions = self.partitions[batch_offsets_uniq].to(device)

        D = torch.index_select(self.view, 0, batch_offsets_uniq)  # , out=buffer[:batch_offsets_uniq.size(0)])
        D = D.to(device)

        D = _decompress(D, D_partitions, self.centroids)
        D = D[batch_offsets_expand.to(device)]

        return D.to(device)


def _decompress(codes, partitions, centroids):
    nvecs, ncentroids, sub_ncentroids, _ = centroids.size()

    codes_ = codes.to(torch.long) + partitions.to(torch.long).unsqueeze(-1)
    centroids_ = centroids.view(nvecs, ncentroids * sub_ncentroids, centroids.size(-1))

    part = _2d_index(centroids_, codes_.T)
    part = part.transpose(-2, 0)
    part = part.view(*part.size()[:-2], -1).contiguous()
    part = torch.nn.functional.normalize(part, dim=-1)

    return part


def _2d_index(embeddings, positions, assign=None):
    bsize, maxlen, *_ = embeddings.size()
    bsize_, *_ = positions.size()

    assert bsize == bsize_, (embeddings.size(), positions.size())
    assert positions.max() < maxlen, (embeddings.size(), positions.size(), positions.max(), maxlen)

    embeddings = embeddings.view(bsize * maxlen, *embeddings.size()[2:])

    R = torch.arange(bsize, device=positions.device)

    for _ in range(len(positions.size()) - 1):
        R = R.unsqueeze(-1)

    positions = positions + R * maxlen

    return embeddings[positions]
