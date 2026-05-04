import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Attach binary labels to a ranked list using qrels.')
    parser.add_argument('--ranking', dest='ranking', required=True,
                        help='Input ranked list in qid<TAB>pid<TAB>rank format.')
    parser.add_argument('--qrels', dest='qrels', required=True,
                        help='Input qrels in qid<TAB>0<TAB>pid<TAB>1 format.')
    parser.add_argument('--output', dest='output', required=True,
                        help='Output ranking in qid<TAB>pid<TAB>rank<TAB>label format.')
    return parser.parse_args()


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            qid, _, pid, rel = line.strip().split('\t')
            if int(rel) > 0:
                qrels[int(qid)].add(int(pid))
    return qrels


def main():
    args = parse_args()
    qrels = load_qrels(args.qrels)

    with open(args.ranking, mode='r', encoding='utf-8') as input_file, \
            open(args.output, mode='w', encoding='utf-8') as output_file:
        for line in input_file:
            qid, pid, rank, *rest = line.strip().split('\t')
            qid = int(qid)
            pid = int(pid)
            label = 1 if pid in qrels.get(qid, set()) else 0
            output_file.write(f'{qid}\t{pid}\t{rank}\t{label}\n')

    print({'output': args.output})


if __name__ == '__main__':
    main()
