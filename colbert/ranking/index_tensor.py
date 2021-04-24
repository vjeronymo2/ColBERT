import torch


class IndexTensor():
    def __init__(self):
        pass

    def from_scratch(self, num_parts, num_embeddings, dim):
        self.offset = 0
        self.tensor = torch.zeros(num_embeddings + 512, dim, dtype=torch.float16)

        self.device = self.tensor.device
        self.dtype = self.tensor.dtype

    def from_tensor(self, tensor):
        self.tensor = tensor
        self.device = tensor.device
        self.dtype = tensor.dtype

    def ingest(self, subtensor):
        endpos = self.offset + subtensor.size(0)
        self.tensor[self.offset : endpos] = subtensor
        self.offset = endpos

    def slice(self, offset, endpos, device=None):
        tensor = self.tensor[offset:endpos + 512]

        if device is not None:
            tensor = tensor.to(device)

        index_tensor = IndexTensor()
        index_tensor.from_tensor(tensor)

        return index_tensor

    def view(self, stride):
        tensor = self.tensor

        dim = tensor.size(-1)
        outdim = tensor.size(0) - stride + 1

        assert dim == 128, "TODO: Remove this temporary assertion!"
        view_tensor = torch.as_strided(tensor, (outdim, stride, dim), (dim, dim, 1))

        return IndexPartView(view_tensor)

    def raw(self):
        return self.tensor


class IndexPartView():
    def __init__(self, view):
        self.view = view
        self.device = view.device

    def select(self, batch_offsets, device, buffer):
        """
            Here, `batch_offsets` has the offsets to the correct documents in the flattend matrix, shifted down
            appropriately (e.g., if there are multiple index parts).

            Returns a PyTorch tensor on the correct device.
        """

        batch_offsets = batch_offsets.to(self.device)
        batch_offsets_uniq, batch_offsets_expand = torch.unique_consecutive(batch_offsets, return_inverse=True)

        D = torch.index_select(self.view, 0, batch_offsets_uniq)  # , out=buffer[:batch_offsets_uniq.size(0)])
        D = D.to(device)
        D = D[batch_offsets_expand.to(device)]

        return D


# TODO: Use the buffer provided above! 
