import argparse
import math
import re
from collections import Counter, defaultdict


TOKEN_PATTERN = re.compile(r'[A-Za-z0-9]+')


def parse_args():
    parser = argparse.ArgumentParser(description='A lightweight BM25 retriever for small-to-medium ColBERT datasets.')
    parser.add_argument('--collection', dest='collection', required=True)
    parser.add_argument('--queries', dest='queries', required=True)
    parser.add_argument('--output', dest='output', required=True)
    parser.add_argument('--topk', dest='topk', type=int, default=1000)
    parser.add_argument('--k1', dest='k1', type=float, default=0.9)
    parser.add_argument('--b', dest='b', type=float, default=0.4)
    parser.add_argument('--title-weight', dest='title_weight', type=float, default=2.0)
    return parser.parse_args()


def tokenize(text):
    return [token.lower() for token in TOKEN_PATTERN.findall(text or '')]


def load_collection(path, title_weight):
    documents = []
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            text = parts[1]
            title = parts[2] if len(parts) >= 3 else ''

            tokens = tokenize(text)
            title_tokens = tokenize(title)
            weighted_tokens = tokens + (title_tokens * max(1, int(round(title_weight))))
            documents.append((pid, weighted_tokens))

    return documents


def load_queries(path):
    queries = []
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            qid, query = line.rstrip('\n').split('\t', 1)
            queries.append((int(qid), tokenize(query)))
    return queries


def build_bm25_index(documents):
    postings = defaultdict(list)
    doc_lengths = {}
    df = Counter()

    for pid, tokens in documents:
        term_counts = Counter(tokens)
        doc_lengths[pid] = len(tokens)
        for term, tf in term_counts.items():
            postings[term].append((pid, tf))
        for term in term_counts:
            df[term] += 1

    avgdl = sum(doc_lengths.values()) / max(1, len(doc_lengths))
    return postings, doc_lengths, df, avgdl


def bm25_score(query_tokens, postings, doc_lengths, df, avgdl, num_docs, k1, b):
    scores = defaultdict(float)
    query_terms = Counter(query_tokens)

    for term, query_tf in query_terms.items():
        if term not in postings:
            continue

        doc_freq = df[term]
        idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        for pid, tf in postings[term]:
            dl = doc_lengths[pid]
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            score = idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
            scores[pid] += score * query_tf

    return scores


def main():
    args = parse_args()

    documents = load_collection(args.collection, args.title_weight)
    queries = load_queries(args.queries)
    postings, doc_lengths, df, avgdl = build_bm25_index(documents)
    num_docs = len(doc_lengths)

    with open(args.output, mode='w', encoding='utf-8') as output_file:
        for qid, query_tokens in queries:
            scores = bm25_score(query_tokens, postings, doc_lengths, df, avgdl, num_docs, args.k1, args.b)
            ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:args.topk]

            for rank, (pid, _) in enumerate(ranking, start=1):
                output_file.write(f'{qid}\t{pid}\t{rank}\n')

    print({'queries': len(queries), 'documents': num_docs, 'output': args.output})


if __name__ == '__main__':
    main()
