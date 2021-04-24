import torch

from itertools import accumulate
from colbert.parameters import DEVICE
from colbert.utils.utils import print_message, dotdict, flatten


class IndexRanker():
    def __init__(self, args, tensor, doclens, bsize=None):
        self.colbert = args.colbert
        self.tensor = tensor
        self.doclens = doclens

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens))

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = args.dim
        self.index_compressed = args.index_compressed

        self.strides = [torch_percentile(self.doclens, p) for p in [90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        print_message(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)

        self.bsize = bsize if bsize else (1 << 14)
        self.buffers = self._create_buffers(self.bsize, self.tensor.dtype, {'cpu', f'cuda:{max(0, args.rank)}'})

    def _create_views(self, tensor):
        views = []

        for stride in self.strides:
            view = tensor.view(stride)
            views.append(view)

        return views

    def _create_buffers(self, max_bsize, dtype, devices):
        buffers = {}

        for device in devices:
            buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
                                           device=device, pin_memory=(device == 'cpu'))
                               for stride in self.strides]

        return buffers

    def rank(self, Q, pids, views=None, shift=0):
        with torch.no_grad():
            assert len(pids) > 0
            assert Q.size(0) in [1, len(pids)]

            Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)

            views = self.views if views is None else views
            VIEWS_DEVICE = views[0].device
            D_buffers = self.buffers[str(VIEWS_DEVICE)]

            raw_pids = pids if type(pids) is list else pids.tolist()
            pids = torch.tensor(pids) if type(pids) is list else pids

            doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

            assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

            one_to_n = torch.arange(len(raw_pids))
            output_pids, output_scores, output_permutation = [], [], []

            for group_idx, stride in enumerate(self.strides):
                locator = (assignments == group_idx)

                if locator.sum() < 1e-5:
                    assert locator.sum() == 0
                    continue

                group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
                group_Q = Q if Q.size(0) == 1 else Q[locator]

                group_offsets = group_offsets - shift

                D = views[group_idx].select(group_offsets, DEVICE, D_buffers[group_idx])
                D = D.to(dtype=self.maxsim_dtype)

                mask = torch.arange(stride, device=DEVICE) + 1
                mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)

                scores = (D @ group_Q) * mask.unsqueeze(-1)
                scores = self.colbert.reduce_matrix(scores, doc_dim=1)

                output_pids.append(group_pids)
                output_scores.append(scores)
                output_permutation.append(one_to_n[locator])

            # TODO: Just allocate zeros above and do output_scores[locator] = X
            output_permutation = torch.cat(output_permutation).sort().indices
            output_pids = torch.cat(output_pids)[output_permutation].tolist()
            output_scores = torch.cat(output_scores)[output_permutation].tolist()

            assert len(raw_pids) == len(output_pids)
            assert len(raw_pids) == len(output_scores)
            assert raw_pids == output_pids

            return output_scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        scores = []
        range_start, range_end = 0, 0

        PIDS_BSIZE = 100_000 if self.index_compressed else 10_000

        for pid_offset in range(0, len(self.doclens), PIDS_BSIZE):
            pid_endpos = min(pid_offset + PIDS_BSIZE, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print_message(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print_message(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor.slice(tensor_offset, tensor_endpos, DEVICE)
            views = self._create_views(collection)

            print_message(f"#> Ranking in batches of {self.bsize} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), self.bsize)):
                if batch_idx % 100 == 0:
                    print_message("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + self.bsize
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

            views = None

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100+1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()
