import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Score a ColBERT ranking.tsv file against qrels.')
    parser.add_argument('--ranking', dest='ranking', required=True)
    parser.add_argument('--qrels', dest='qrels', required=True)
    parser.add_argument('--depth', dest='depth', type=int, default=1000)
    return parser.parse_args()


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            qid, _, pid, rel = line.strip().split('\t')
            if int(rel) > 0:
                qrels[int(qid)].add(int(pid))
    return qrels


def load_ranking(path, depth):
    ranking = defaultdict(list)
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            qid, pid, rank, *rest = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)
            if rank <= depth:
                ranking[qid].append((rank, pid))

    for qid in ranking:
        ranking[qid] = [pid for _, pid in sorted(ranking[qid])]
    return ranking


def main():
    args = parse_args()
    qrels = load_qrels(args.qrels)
    ranking = load_ranking(args.ranking, args.depth)

    metrics = {
        'MRR@10': 0.0,
        'Recall@100': 0.0,
        'Recall@1000': 0.0,
        'Success@10': 0.0,
    }

    num_queries = len(qrels)
    for qid, gold in qrels.items():
        preds = ranking.get(qid, [])
        first_hit = None
        for idx, pid in enumerate(preds[:10]):
            if pid in gold:
                first_hit = idx + 1
                break

        if first_hit is not None:
            metrics['MRR@10'] += 1.0 / first_hit
            metrics['Success@10'] += 1.0

        metrics['Recall@100'] += len([pid for pid in preds[:100] if pid in gold]) / max(1, len(gold))
        metrics['Recall@1000'] += len([pid for pid in preds[:1000] if pid in gold]) / max(1, len(gold))

    for key in metrics:
        metrics[key] /= max(1, num_queries)

    print(metrics)


if __name__ == '__main__':
    main()
