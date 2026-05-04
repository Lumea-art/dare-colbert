import os
import torch

from colbert.ranking.index_part import IndexPart
from colbert.ranking.faiss_index import FaissIndex
from colbert.indexing.faiss import get_faiss_index_name
from colbert.indexing.manifest import list_active_segments


class SegmentRanker():
    def __init__(self, segment_name, index_path, faiss_index_path, nprobe, dim, part_range=None, verbose=True):
        self.segment_name = segment_name
        self.index_path = index_path
        self.faiss_index = None if faiss_index_path is None else FaissIndex(index_path, faiss_index_path, nprobe, part_range=part_range)
        self.index = IndexPart(index_path, dim=dim, part_range=part_range, verbose=verbose)

    def retrieve(self, faiss_depth, Q, verbose=False):
        if self.faiss_index is None:
            raise ValueError(f"Segment {self.segment_name} has no FAISS index attached.")
        return self.faiss_index.retrieve(faiss_depth, Q, verbose=verbose)

    def contains_pid(self, pid):
        return self.index.pid_in_range(pid)

    def score(self, Q, pids):
        return self.index.rank(Q, pids)


class Ranker():
    def __init__(self, args, inference, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth
        self.segments = []

        default_faiss_name = args.faiss_name if getattr(args, 'faiss_name', None) is not None else get_faiss_index_name(args)
        manifest_segments = list_active_segments(args.index_path, default_faiss_name=default_faiss_name)

        for segment in manifest_segments:
            faiss_name = segment.get('faiss_name')
            segment_index_path = args.index_path if segment.get('relative_path', '.') == '.' \
                else os.path.join(args.index_path, segment['relative_path'])
            if not os.path.exists(segment_index_path):
                continue
            faiss_index_path = None if faiss_name is None else os.path.join(segment_index_path, faiss_name)

            if faiss_depth is not None and (faiss_index_path is None or not os.path.exists(faiss_index_path)):
                continue

            self.segments.append(SegmentRanker(
                segment_name=segment['name'],
                index_path=segment_index_path,
                faiss_index_path=faiss_index_path,
                nprobe=args.nprobe,
                dim=inference.colbert.dim,
                part_range=args.part_range,
                verbose=(segment.get('relative_path', '.') == '.'),
            ))

    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None)

        return Q

    def retrieve_candidates(self, Q, verbose=False):
        assert len(self.segments) > 0, "No active index segments were found."

        if self.faiss_depth is None:
            raise ValueError("Retrieval requires a valid FAISS depth.")

        merged = [[] for _ in range(Q.size(0))]
        seen = [set() for _ in range(Q.size(0))]

        for segment in self.segments:
            all_pids = segment.retrieve(self.faiss_depth, Q, verbose=verbose)
            for query_idx, pids in enumerate(all_pids):
                for pid in pids:
                    if pid not in seen[query_idx]:
                        seen[query_idx].add(pid)
                        merged[query_idx].append(pid)

        return merged

    def rank(self, Q, pids=None):
        pids = self.retrieve_candidates(Q, verbose=False)[0] if pids is None else pids

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        score_pairs = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1)
            for segment in self.segments:
                segment_pids = [pid for pid in pids if segment.contains_pid(pid)]
                if len(segment_pids) == 0:
                    continue

                segment_scores = segment.score(Q, segment_pids)
                score_pairs.extend(zip(segment_pids, segment_scores))

            if len(score_pairs) > 0:
                score_pairs = sorted(score_pairs, key=lambda item: item[1], reverse=True)
                pids = [pid for pid, _ in score_pairs]
                scores = [score for _, score in score_pairs]
            else:
                pids, scores = [], []
        else:
            scores = []

        return pids, scores
