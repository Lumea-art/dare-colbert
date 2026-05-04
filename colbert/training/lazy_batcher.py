import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.utils.metadata import load_collection_meta, default_doc_meta

from colbert.utils.runs import Run


class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = self._load_triples(args.triples, rank, nranks)
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)
        self.collection_meta = self._load_collection_meta(args.collection_meta, len(self.collection))

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    qid, pos, neg = ujson.loads(line)
                    triples.append((qid, pos, neg))

        return triples

    def _load_queries(self, path):
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query

        return queries

    def _load_collection(self, path):
        print_message("#> Loading collection...")

        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                pid, passage, title, *_ = line.strip().split('\t')
                assert pid == 'id' or int(pid) == line_idx

                passage = title + ' | ' + passage
                collection.append(passage)

        return collection

    def _load_collection_meta(self, path, expected_size):
        print_message("#> Loading collection metadata...")
        metadata = load_collection_meta(path, expected_size=expected_size)

        if len(metadata) < expected_size:
            metadata.extend(default_doc_meta(pid) for pid in range(len(metadata), expected_size))

        return metadata

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        queries, positives, negatives = [], [], []
        positives_meta, negatives_meta = [], []

        for position in range(offset, endpos):
            query, pos, neg = self.triples[position]
            query, pos, neg = self.queries[query], self.collection[pos], self.collection[neg]

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
            positives_meta.append(self.collection_meta[pos])
            negatives_meta.append(self.collection_meta[neg])

        return self.collate(queries, positives, negatives, positives_meta, negatives_meta)

    def collate(self, queries, positives, negatives, positive_metadata=None, negative_metadata=None):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(
            queries,
            positives,
            negatives,
            self.bsize // self.accumsteps,
            positive_metadata=positive_metadata,
            negative_metadata=negative_metadata,
        )

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
