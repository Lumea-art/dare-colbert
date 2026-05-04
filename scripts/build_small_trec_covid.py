import argparse
import json
import os
import random
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a smaller, reproducible TREC-COVID incremental dataset from a processed base+delta root.'
    )
    parser.add_argument('--input-root', dest='input_root', required=True,
                        help='Input processed incremental dataset root, e.g. data/processed/trec_covid')
    parser.add_argument('--output-dir', dest='output_dir', required=True,
                        help='Output directory for the reduced dataset, e.g. data/processed/trec_covid_small')
    parser.add_argument('--rounds', dest='rounds', nargs='+', type=int, default=None,
                        help='Rounds to keep. Defaults to all rounds from the input manifest.')
    parser.add_argument('--max-base-docs', dest='max_base_docs', type=int, default=None,
                        help='Maximum number of documents in the reduced base segment.')
    parser.add_argument('--max-delta-docs', dest='max_delta_docs', type=int, default=None,
                        help='Maximum number of documents per reduced delta segment.')
    parser.add_argument('--max-queries-per-round', dest='max_queries_per_round', type=int, default=None,
                        help='Maximum number of queries to keep per round. Defaults to all queries with qrels.')
    parser.add_argument('--seed', dest='seed', type=int, default=13,
                        help='Random seed for reproducible downsampling.')
    parser.add_argument('--skip-meta', dest='skip_meta', default=False, action='store_true',
                        help='Do not write collection_meta.jsonl even if the source dataset has metadata.')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, mode='r', encoding='utf-8') as input_file:
        return json.load(input_file)


def resolve_relative(root, relative_path):
    return os.path.normpath(os.path.join(root, relative_path))


def ordered_round_ids(manifest):
    return sorted(int(round_id) for round_id in manifest.get('rounds', {}).keys())


def select_rounds(manifest, requested_rounds):
    available = ordered_round_ids(manifest)
    if requested_rounds is None:
        return available

    requested = sorted(set(requested_rounds))
    missing = [round_id for round_id in requested if round_id not in available]
    if missing:
        raise ValueError(f'Requested rounds are missing from manifest: {missing}. Available rounds: {available}')

    return requested


def read_collection(path):
    rows = []
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.rstrip('\n')
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                raise ValueError(f'Malformed collection row in {path}:{line_number}')

            pid = int(parts[0])
            text = parts[1]
            title = parts[2] if len(parts) >= 3 else ''
            rows.append({
                'pid': pid,
                'text': text,
                'title': title,
            })

    return rows


def read_meta(path):
    metadata = {}
    if path is None or not os.path.exists(path):
        return metadata

    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            metadata[int(item['pid'])] = item

    return metadata


def read_queries(path):
    queries = OrderedDict()
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.rstrip('\n')
            if not line:
                continue
            qid, query = line.split('\t', 1)
            queries[int(qid)] = query
    return queries


def read_qrels(path):
    qrels = defaultdict(list)
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            qid, _, pid, rel = line.split('\t')
            rel = int(rel)
            if rel > 0:
                qrels[int(qid)].append((int(pid), rel))
    return qrels


def read_pid_lookup(path):
    if not os.path.exists(path):
        return {}

    lookup = {}
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lookup[int(item['pid'])] = item
    return lookup


def segment_paths(root, manifest, source_round):
    base_round = int(manifest['base_round'])
    if source_round == base_round:
        return {
            'collection': os.path.join(root, 'base', 'collection.tsv'),
            'meta': os.path.join(root, 'base', 'collection_meta.jsonl'),
        }

    round_info = manifest['rounds'][str(source_round)]
    return {
        'collection': resolve_relative(root, round_info['delta_collection_path']),
        'meta': None if round_info.get('delta_meta_path') is None else resolve_relative(root, round_info['delta_meta_path']),
    }


def load_segment_docs(root, manifest, round_ids):
    docs_by_first_seen_round = OrderedDict()

    for round_id in round_ids:
        paths = segment_paths(root, manifest, round_id)
        collection_rows = read_collection(paths['collection'])
        meta_rows = read_meta(paths['meta'])

        segment_docs = []
        for row in collection_rows:
            meta = dict(meta_rows.get(row['pid'], {}))
            meta.setdefault('pid', row['pid'])
            meta.setdefault('first_seen_round', round_id)
            segment_docs.append({
                'old_pid': row['pid'],
                'text': row['text'],
                'title': row['title'],
                'meta': meta,
                'first_seen_round': round_id,
            })

        docs_by_first_seen_round[round_id] = segment_docs

    return docs_by_first_seen_round


def choose_queries(queries, qrels, max_queries, seed):
    eligible_qids = [qid for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
    if max_queries is None or len(eligible_qids) <= max_queries:
        chosen = set(eligible_qids)
    else:
        rng = random.Random(seed)
        chosen = set(rng.sample(eligible_qids, max_queries))

    selected_queries = OrderedDict((qid, query) for qid, query in queries.items() if qid in chosen)
    selected_qrels = {qid: list(qrels[qid]) for qid in selected_queries.keys()}
    return selected_queries, selected_qrels


def collect_required_docs(rounds, chosen_qrels_by_round, first_seen_lookup):
    required = defaultdict(set)

    for round_id in rounds:
        qrels = chosen_qrels_by_round[round_id]
        for pairs in qrels.values():
            for old_pid, _ in pairs:
                if old_pid not in first_seen_lookup:
                    continue
                required[first_seen_lookup[old_pid]].add(old_pid)

    return required


def sample_segment_docs(candidate_docs, required_pids, quota, seed, label):
    candidate_pids = [doc['old_pid'] for doc in candidate_docs]
    candidate_pid_set = set(candidate_pids)
    required_pids = set(pid for pid in required_pids if pid in candidate_pid_set)

    if quota is None or quota >= len(candidate_docs):
        return list(candidate_docs)

    if len(required_pids) > quota:
        raise ValueError(
            f'Segment {label} needs at least {len(required_pids)} qrel-positive docs, '
            f'but quota is only {quota}. Increase the quota or reduce sampled queries.'
        )

    remaining = [pid for pid in candidate_pids if pid not in required_pids]
    rng = random.Random(seed)
    extra_needed = quota - len(required_pids)
    sampled_extra = set(rng.sample(remaining, extra_needed)) if extra_needed > 0 else set()
    keep = required_pids | sampled_extra

    return [doc for doc in candidate_docs if doc['old_pid'] in keep]


def reassign_pids(output_segments):
    old_to_new = {}
    next_pid = 0

    for _, docs in output_segments:
        for doc in docs:
            old_to_new[doc['old_pid']] = next_pid
            next_pid += 1

    return old_to_new


def write_collection(path, docs, old_to_new):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for doc in docs:
            output_file.write(f"{old_to_new[doc['old_pid']]}\t{doc['text']}\t{doc['title']}\n")


def write_meta(path, docs, old_to_new):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for doc in docs:
            payload = dict(doc['meta'])
            payload['pid'] = old_to_new[doc['old_pid']]
            output_file.write(json.dumps(payload, ensure_ascii=False) + '\n')


def write_queries(path, queries):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for qid, query in queries.items():
            output_file.write(f'{qid}\t{query}\n')


def write_qrels(path, qrels, old_to_new):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for qid in sorted(qrels.keys()):
            for old_pid, rel in qrels[qid]:
                output_file.write(f'{qid}\t0\t{old_to_new[old_pid]}\t{rel}\n')


def write_cumulative_docids(path, cumulative_docs, pid_lookup):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for doc in cumulative_docs:
            old_pid = doc['old_pid']
            docid = pid_lookup.get(old_pid, {}).get('cord_uid', str(old_pid))
            output_file.write(f'{docid}\n')


def main():
    args = parse_args()

    manifest = load_json(os.path.join(args.input_root, 'manifest.json'))
    selected_rounds = select_rounds(manifest, args.rounds)
    pid_lookup = read_pid_lookup(os.path.join(args.input_root, 'pid_lookup.jsonl'))
    source_rounds = ordered_round_ids(manifest)
    max_selected_round = selected_rounds[-1]

    source_docs_by_round = load_segment_docs(
        args.input_root,
        manifest,
        [round_id for round_id in source_rounds if round_id <= max_selected_round],
    )

    first_seen_lookup = {}
    for round_id, docs in source_docs_by_round.items():
        for doc in docs:
            first_seen_lookup[doc['old_pid']] = round_id

    chosen_queries_by_round = {}
    chosen_qrels_by_round = {}

    for round_id in selected_rounds:
        round_info = manifest['rounds'][str(round_id)]
        queries = read_queries(resolve_relative(args.input_root, round_info['queries_path']))
        qrels = read_qrels(resolve_relative(args.input_root, round_info['qrels_path']))
        selected_queries, selected_qrels = choose_queries(
            queries,
            qrels,
            args.max_queries_per_round,
            seed=args.seed + round_id,
        )
        chosen_queries_by_round[round_id] = selected_queries
        chosen_qrels_by_round[round_id] = selected_qrels

    required_docs = collect_required_docs(selected_rounds, chosen_qrels_by_round, first_seen_lookup)

    new_base_round = selected_rounds[0]
    base_candidates = []
    for round_id in source_rounds:
        if round_id > new_base_round:
            break
        base_candidates.extend(source_docs_by_round[round_id])

    base_required = set()
    for round_id in source_rounds:
        if round_id > new_base_round:
            break
        base_required.update(required_docs.get(round_id, set()))

    output_segments = []
    base_docs = sample_segment_docs(
        base_candidates,
        base_required,
        args.max_base_docs,
        seed=args.seed + 1000 + new_base_round,
        label=f'base<=round{new_base_round}',
    )
    output_segments.append((new_base_round, base_docs))

    for round_id in selected_rounds[1:]:
        delta_candidates = list(source_docs_by_round.get(round_id, []))
        delta_docs = sample_segment_docs(
            delta_candidates,
            required_docs.get(round_id, set()),
            args.max_delta_docs,
            seed=args.seed + 2000 + round_id,
            label=f'delta-round{round_id}',
        )
        output_segments.append((round_id, delta_docs))

    old_to_new = reassign_pids(output_segments)
    kept_old_pids = set(old_to_new.keys())

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, 'base'))
    ensure_dir(os.path.join(args.output_dir, 'delta'))
    ensure_dir(os.path.join(args.output_dir, 'rounds'))

    output_manifest = {
        'dataset': manifest.get('dataset', 'trec-covid-incremental') + '-small',
        'source_incremental_root': os.path.abspath(args.input_root),
        'base_round': new_base_round,
        'rounds': OrderedDict(),
        'num_documents': len(old_to_new),
        'topic_field': manifest.get('topic_field'),
        'docids_mode': manifest.get('docids_mode'),
        'sampling': {
            'seed': args.seed,
            'max_base_docs': args.max_base_docs,
            'max_delta_docs': args.max_delta_docs,
            'max_queries_per_round': args.max_queries_per_round,
        },
    }

    write_collection(os.path.join(args.output_dir, 'base', 'collection.tsv'), base_docs, old_to_new)
    meta_written = False
    if not args.skip_meta:
        write_meta(os.path.join(args.output_dir, 'base', 'collection_meta.jsonl'), base_docs, old_to_new)
        meta_written = True

    cumulative_docs = []
    cumulative_old_pid_set = set()

    for segment_round, docs in output_segments:
        cumulative_docs.extend(docs)
        cumulative_old_pid_set.update(doc['old_pid'] for doc in docs)

        if segment_round != new_base_round:
            round_delta_dir = os.path.join(args.output_dir, 'delta', f'round{segment_round}')
            ensure_dir(round_delta_dir)
            write_collection(os.path.join(round_delta_dir, 'collection.tsv'), docs, old_to_new)
            if not args.skip_meta:
                write_meta(os.path.join(round_delta_dir, 'collection_meta.jsonl'), docs, old_to_new)

        round_queries_out = OrderedDict()
        round_qrels_out = defaultdict(list)

        selected_queries = chosen_queries_by_round[segment_round]
        selected_qrels = chosen_qrels_by_round[segment_round]

        for qid, query in selected_queries.items():
            filtered_pairs = [(old_pid, rel) for old_pid, rel in selected_qrels.get(qid, []) if old_pid in cumulative_old_pid_set]
            if len(filtered_pairs) == 0:
                continue
            round_queries_out[qid] = query
            round_qrels_out[qid].extend(filtered_pairs)

        round_root = os.path.join(args.output_dir, 'rounds', f'round{segment_round}')
        ensure_dir(round_root)
        write_queries(os.path.join(round_root, 'queries.tsv'), round_queries_out)
        write_qrels(os.path.join(round_root, 'qrels.tsv'), round_qrels_out, old_to_new)
        write_cumulative_docids(os.path.join(round_root, 'cumulative_docids.txt'), cumulative_docs, pid_lookup)

        delta_count = 0 if segment_round == new_base_round else len(docs)
        output_manifest['rounds'][str(segment_round)] = {
            'num_queries': len(round_queries_out),
            'num_qrels': sum(len(values) for values in round_qrels_out.values()),
            'num_cumulative_docs': len(cumulative_docs),
            'num_delta_docs': delta_count,
            'queries_path': os.path.relpath(os.path.join(round_root, 'queries.tsv'), args.output_dir),
            'qrels_path': os.path.relpath(os.path.join(round_root, 'qrels.tsv'), args.output_dir),
            'delta_collection_path': None if segment_round == new_base_round else os.path.relpath(
                os.path.join(args.output_dir, 'delta', f'round{segment_round}', 'collection.tsv'),
                args.output_dir,
            ),
            'delta_meta_path': None if (segment_round == new_base_round or args.skip_meta) else os.path.relpath(
                os.path.join(args.output_dir, 'delta', f'round{segment_round}', 'collection_meta.jsonl'),
                args.output_dir,
            ),
        }

    output_manifest['num_sources'] = len({
        int(doc['meta'].get('source_id', 0))
        for _, docs in output_segments
        for doc in docs
    })

    if meta_written:
        pass

    with open(os.path.join(args.output_dir, 'manifest.json'), mode='w', encoding='utf-8') as output_file:
        json.dump(output_manifest, output_file, indent=2)
        output_file.write('\n')

    with open(os.path.join(args.output_dir, 'pid_lookup.jsonl'), mode='w', encoding='utf-8') as output_file:
        for _, docs in output_segments:
            for doc in docs:
                old_pid = doc['old_pid']
                source_info = dict(pid_lookup.get(old_pid, {}))
                payload = {
                    'pid': old_to_new[old_pid],
                    'original_pid': old_pid,
                    'first_seen_round': doc['first_seen_round'],
                    'source_id': int(doc['meta'].get('source_id', source_info.get('source_id', 0))),
                    'source_name': doc['meta'].get('source_name', source_info.get('source_name', '')),
                    'publish_time': doc['meta'].get('publish_time', source_info.get('publish_time', '')),
                    'cord_uid': source_info.get('cord_uid', str(old_pid)),
                    'title': source_info.get('title', doc['title']),
                }
                output_file.write(json.dumps(payload, ensure_ascii=False) + '\n')

    print(json.dumps({
        'output_dir': args.output_dir,
        'base_round': new_base_round,
        'num_documents': len(old_to_new),
        'rounds': output_manifest['rounds'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
