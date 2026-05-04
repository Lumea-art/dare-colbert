import argparse
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare a 4090-friendly incremental TREC-COVID dataset for ColBERT.')

    parser.add_argument('--metadata-csv', dest='metadata_csv', required=True,
                        help='Path to the CORD-19 metadata.csv file.')
    parser.add_argument('--docids-dir', dest='docids_dir', required=True,
                        help='Directory containing docids-rnd{round}.txt files.')
    parser.add_argument('--topics-dir', dest='topics_dir', required=True,
                        help='Directory containing topics-rnd{round}.xml or .txt files.')
    parser.add_argument('--qrels-dir', dest='qrels_dir', required=True,
                        help='Directory containing qrels-covid_d5_j0.5-rnd{round}.txt or qrels-rnd{round}.txt files.')
    parser.add_argument('--output-dir', dest='output_dir', required=True,
                        help='Directory to write the processed dataset.')

    parser.add_argument('--rounds', dest='rounds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Rounds to prepare, e.g. --rounds 1 2 3 4 5')
    parser.add_argument('--docids-pattern', dest='docids_pattern', default='docids-rnd{round}.txt')
    parser.add_argument('--topics-pattern', dest='topics_pattern', default='topics-rnd{round}.xml')
    parser.add_argument('--qrels-pattern', dest='qrels_pattern', default='qrels-covid_d5_j0.5-rnd{round}.txt')

    parser.add_argument('--topic-field', dest='topic_field', default='query',
                        choices=['query', 'question', 'query+question'],
                        help='Which topic field to keep as the retrieval query.')
    parser.add_argument('--max-abstract-chars', dest='max_abstract_chars', type=int, default=4000,
                        help='Soft truncate long abstracts for manageable passages.')
    parser.add_argument('--min-text-chars', dest='min_text_chars', type=int, default=40,
                        help='Discard documents with less than this many characters after cleanup.')
    parser.add_argument('--fallback-body-root', dest='fallback_body_root', default=None,
                        help='Optional root directory for parsing body JSON when abstract is missing.')
    parser.add_argument('--source-split-regex', dest='source_split_regex', default=r'[;,|]',
                        help='Regex for splitting source_x into source names.')
    parser.add_argument('--time-bucket-mode', dest='time_bucket_mode', default='round',
                        choices=['round', 'month'],
                        help='How to derive time_bucket_id for collection_meta.jsonl.')
    parser.add_argument('--docids-mode', dest='docids_mode', default='cumulative',
                        choices=['cumulative', 'incremental'],
                        help='Whether each docids-rnd file already contains the full cumulative collection '
                             'up to that round, or only the newly added documents.')

    return parser.parse_args()


def read_docids(path):
    docids = []
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            docid = line.strip()
            if docid:
                docids.append(docid)
    return docids


def parse_topics(path, topic_field):
    if path.endswith('.xml'):
        return parse_topics_xml(path, topic_field)
    if path.endswith('.json') or path.endswith('.jsonl'):
        return parse_topics_json(path, topic_field)
    return parse_topics_tsv(path)


def parse_topics_xml(path, topic_field):
    root = ET.parse(path).getroot()
    queries = OrderedDict()

    for topic in root.findall('.//topic'):
        qid = int(topic.attrib.get('number', topic.findtext('number')).strip())
        query = (topic.findtext('query') or '').strip()
        question = (topic.findtext('question') or '').strip()

        if topic_field == 'query':
            text = query
        elif topic_field == 'question':
            text = question or query
        else:
            text = ' '.join([part for part in [query, question] if part])

        if text:
            queries[qid] = normalize_whitespace(text)

    return queries


def parse_topics_json(path, topic_field):
    queries = OrderedDict()

    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = int(item.get('number', item.get('qid', item.get('id'))))
            query = normalize_whitespace(item.get('query', ''))
            question = normalize_whitespace(item.get('question', ''))

            if topic_field == 'query':
                text = query
            elif topic_field == 'question':
                text = question or query
            else:
                text = ' '.join([part for part in [query, question] if part])

            if text:
                queries[qid] = text

    return queries


def parse_topics_tsv(path):
    queries = OrderedDict()
    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.rstrip('\n')
            if not line:
                continue
            qid, query = line.split('\t', 1)
            queries[int(qid)] = normalize_whitespace(query)
    return queries


def parse_qrels(path):
    qrels = defaultdict(list)

    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            qid = int(qid)
            rel = int(rel)
            if rel > 0:
                qrels[qid].append(docid)

    return qrels


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', (text or '')).strip()


def parse_publish_time(raw_value):
    raw_value = (raw_value or '').strip()
    if not raw_value:
        return None

    for pattern in ('%Y-%m-%d', '%Y-%m', '%Y'):
        try:
            parsed = datetime.strptime(raw_value, pattern)
            if pattern == '%Y':
                parsed = parsed.replace(month=1, day=1)
            elif pattern == '%Y-%m':
                parsed = parsed.replace(day=1)
            return parsed
        except ValueError:
            continue

    return None


def safe_json_load(path):
    with open(path, mode='r', encoding='utf-8') as input_file:
        return json.load(input_file)


def read_body_from_metadata(row, body_root):
    if body_root is None:
        return ''

    candidates = []
    for key in ('pmc_json_files', 'pdf_json_files'):
        value = (row.get(key) or '').strip()
        if value:
            candidates.extend([item.strip() for item in value.split(';') if item.strip()])

    for relative_path in candidates:
        possible_paths = [
            relative_path,
            os.path.join(body_root, relative_path),
            os.path.join(body_root, relative_path.replace('/', os.sep)),
        ]

        for candidate_path in possible_paths:
            if not os.path.exists(candidate_path):
                continue

            try:
                payload = safe_json_load(candidate_path)
                paragraphs = [normalize_whitespace(block.get('text', '')) for block in payload.get('body_text', [])]
                paragraphs = [paragraph for paragraph in paragraphs if paragraph]
                if paragraphs:
                    return ' '.join(paragraphs)
            except Exception:
                continue

    return ''


def load_metadata(args, ordered_docids):
    ordered_docid_set = set(ordered_docids)
    metadata = {}
    source_vocab = OrderedDict()

    with open(args.metadata_csv, mode='r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            docid = (row.get('cord_uid') or '').strip()
            if not docid or docid not in ordered_docid_set:
                continue

            title = normalize_whitespace(row.get('title', ''))
            abstract = normalize_whitespace(row.get('abstract', ''))
            if len(abstract) > args.max_abstract_chars:
                abstract = abstract[:args.max_abstract_chars].rsplit(' ', 1)[0]

            body = ''
            if len(abstract) < args.min_text_chars:
                body = normalize_whitespace(read_body_from_metadata(row, args.fallback_body_root))

            text = abstract if len(abstract) >= args.min_text_chars else body
            if len(text) < args.min_text_chars:
                continue

            publish_time = parse_publish_time(row.get('publish_time', ''))
            source_names = [normalize_whitespace(item) for item in re.split(args.source_split_regex, row.get('source_x', '') or '')]
            source_names = [item for item in source_names if item]
            primary_source = source_names[0] if source_names else 'unknown'
            if primary_source not in source_vocab:
                source_vocab[primary_source] = len(source_vocab)

            metadata[docid] = {
                'cord_uid': docid,
                'title': title or '-',
                'text': text,
                'publish_time': publish_time.strftime('%Y-%m-%d') if publish_time is not None else '',
                'publish_datetime': publish_time,
                'source_name': primary_source,
                'source_id': source_vocab[primary_source],
            }

    missing = [docid for docid in ordered_docids if docid not in metadata]
    if missing:
        print(f'[WARN] Skipping {len(missing)} docids with missing/empty metadata.')

    filtered_docids = [docid for docid in ordered_docids if docid in metadata]
    return metadata, filtered_docids, source_vocab


def derive_time_bucket_id(doc_meta, first_seen_round, args):
    if args.time_bucket_mode == 'round':
        return max(0, first_seen_round - 1)

    publish_time = doc_meta.get('publish_datetime')
    if publish_time is None:
        return max(0, first_seen_round - 1)

    return (publish_time.year * 12) + publish_time.month


def build_round_views(round_to_docids):
    views = {}
    cumulative = []
    cumulative_set = set()

    for round_id in sorted(round_to_docids):
        current = round_to_docids[round_id]
        delta = [docid for docid in current if docid not in cumulative_set]
        cumulative.extend(delta)
        cumulative_set.update(delta)
        views[round_id] = {
            'cumulative': list(cumulative),
            'delta': delta,
        }

    return views


def normalize_round_docids(round_to_docids, docids_mode):
    if docids_mode == 'cumulative':
        return OrderedDict((round_id, list(docids)) for round_id, docids in round_to_docids.items())

    normalized = OrderedDict()
    cumulative = []
    seen = set()
    for round_id in sorted(round_to_docids):
        for docid in round_to_docids[round_id]:
            if docid in seen:
                continue
            seen.add(docid)
            cumulative.append(docid)
        normalized[round_id] = list(cumulative)

    return normalized


def write_collection(path, docids, pid_lookup, metadata):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for docid in docids:
            pid = pid_lookup[docid]
            item = metadata[docid]
            output_file.write(f"{pid}\t{item['text']}\t{item['title']}\n")


def compute_recency_norm(first_seen_round, min_round, max_round):
    if max_round == min_round:
        return 0.0
    return float(first_seen_round - min_round) / float(max_round - min_round)


def write_collection_meta(path, docids, pid_lookup, metadata, doc_first_seen_round, args):
    min_round = min(doc_first_seen_round.values())
    max_round = max(doc_first_seen_round.values())

    with open(path, mode='w', encoding='utf-8') as output_file:
        for docid in docids:
            pid = pid_lookup[docid]
            first_seen_round = doc_first_seen_round[docid]
            item = metadata[docid]
            payload = {
                'pid': pid,
                'time_bucket_id': derive_time_bucket_id(item, first_seen_round, args),
                'source_id': item['source_id'],
                'recency_norm': round(compute_recency_norm(first_seen_round, min_round, max_round), 6),
                'cluster_label': -100,
                'cluster_confidence': 1.0,
                'cord_uid': docid,
                'source_name': item['source_name'],
                'publish_time': item['publish_time'],
                'first_seen_round': first_seen_round,
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + '\n')


def write_queries(path, queries):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for qid, query in queries.items():
            output_file.write(f'{qid}\t{query}\n')


def write_qrels(path, qrels, pid_lookup):
    with open(path, mode='w', encoding='utf-8') as output_file:
        for qid in sorted(qrels):
            for docid in qrels[qid]:
                if docid not in pid_lookup:
                    continue
                output_file.write(f'{qid}\t0\t{pid_lookup[docid]}\t1\n')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_existing(path_candidates, what):
    for candidate in path_candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'Could not locate {what}. Tried: {path_candidates}')


def main():
    args = parse_args()

    rounds = sorted(set(args.rounds))
    ensure_dir(args.output_dir)

    round_to_docids = OrderedDict()
    round_to_queries = OrderedDict()
    round_to_qrels = OrderedDict()

    for round_id in rounds:
        docids_path = resolve_existing(
            [os.path.join(args.docids_dir, args.docids_pattern.format(round=round_id))],
            f'docids file for round {round_id}',
        )
        topics_path = resolve_existing(
            [
                os.path.join(args.topics_dir, args.topics_pattern.format(round=round_id)),
                os.path.join(args.topics_dir, args.topics_pattern.format(round=round_id).replace('.xml', '.txt')),
                os.path.join(args.topics_dir, args.topics_pattern.format(round=round_id).replace('.xml', '.json')),
            ],
            f'topics file for round {round_id}',
        )
        qrels_path = resolve_existing(
            [
                os.path.join(args.qrels_dir, args.qrels_pattern.format(round=round_id)),
                os.path.join(args.qrels_dir, f'qrels-rnd{round_id}.txt'),
            ],
            f'qrels file for round {round_id}',
        )

        round_to_docids[round_id] = read_docids(docids_path)
        round_to_queries[round_id] = parse_topics(topics_path, args.topic_field)
        round_to_qrels[round_id] = parse_qrels(qrels_path)

    round_to_docids = normalize_round_docids(round_to_docids, args.docids_mode)

    ordered_docids = []
    seen = set()
    doc_first_seen_round = {}
    for round_id in rounds:
        for docid in round_to_docids[round_id]:
            if docid in seen:
                continue
            seen.add(docid)
            ordered_docids.append(docid)
            doc_first_seen_round[docid] = round_id

    metadata, ordered_docids, source_vocab = load_metadata(args, ordered_docids)
    pid_lookup = {docid: idx for idx, docid in enumerate(ordered_docids)}

    round_views = build_round_views({
        round_id: [docid for docid in round_to_docids[round_id] if docid in pid_lookup]
        for round_id in rounds
    })

    base_dir = os.path.join(args.output_dir, 'base')
    delta_dir = os.path.join(args.output_dir, 'delta')
    rounds_dir = os.path.join(args.output_dir, 'rounds')
    ensure_dir(base_dir)
    ensure_dir(delta_dir)
    ensure_dir(rounds_dir)

    base_round = rounds[0]
    write_collection(os.path.join(base_dir, 'collection.tsv'), round_views[base_round]['cumulative'], pid_lookup, metadata)
    write_collection_meta(os.path.join(base_dir, 'collection_meta.jsonl'),
                          round_views[base_round]['cumulative'], pid_lookup, metadata, doc_first_seen_round, args)

    manifest = {
        'dataset': 'trec-covid-incremental',
        'base_round': base_round,
        'rounds': {},
        'num_documents': len(ordered_docids),
        'num_sources': len(source_vocab),
        'topic_field': args.topic_field,
        'docids_mode': args.docids_mode,
    }

    for round_id in rounds:
        round_root = os.path.join(rounds_dir, f'round{round_id}')
        ensure_dir(round_root)

        queries = round_to_queries[round_id]
        qrels = round_to_qrels[round_id]

        write_queries(os.path.join(round_root, 'queries.tsv'), queries)
        write_qrels(os.path.join(round_root, 'qrels.tsv'), qrels, pid_lookup)

        cumulative_docids = round_views[round_id]['cumulative']
        delta_docids = round_views[round_id]['delta']

        with open(os.path.join(round_root, 'cumulative_docids.txt'), mode='w', encoding='utf-8') as output_file:
            for docid in cumulative_docids:
                output_file.write(f'{docid}\n')

        if round_id != base_round:
            round_delta_dir = os.path.join(delta_dir, f'round{round_id}')
            ensure_dir(round_delta_dir)
            write_collection(os.path.join(round_delta_dir, 'collection.tsv'), delta_docids, pid_lookup, metadata)
            write_collection_meta(os.path.join(round_delta_dir, 'collection_meta.jsonl'),
                                  delta_docids, pid_lookup, metadata, doc_first_seen_round, args)

        manifest['rounds'][str(round_id)] = {
            'num_queries': len(queries),
            'num_qrels': sum(len(v) for v in qrels.values()),
            'num_cumulative_docs': len(cumulative_docids),
            'num_delta_docs': len(delta_docids),
            'queries_path': os.path.relpath(os.path.join(round_root, 'queries.tsv'), args.output_dir),
            'qrels_path': os.path.relpath(os.path.join(round_root, 'qrels.tsv'), args.output_dir),
            'delta_collection_path': None if round_id == base_round else os.path.relpath(
                os.path.join(delta_dir, f'round{round_id}', 'collection.tsv'), args.output_dir),
            'delta_meta_path': None if round_id == base_round else os.path.relpath(
                os.path.join(delta_dir, f'round{round_id}', 'collection_meta.jsonl'), args.output_dir),
        }

    with open(os.path.join(args.output_dir, 'manifest.json'), mode='w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, indent=2)
        output_file.write('\n')

    with open(os.path.join(args.output_dir, 'pid_lookup.jsonl'), mode='w', encoding='utf-8') as output_file:
        for docid in ordered_docids:
            item = metadata[docid]
            payload = {
                'pid': pid_lookup[docid],
                'cord_uid': docid,
                'title': item['title'],
                'source_name': item['source_name'],
                'source_id': item['source_id'],
                'publish_time': item['publish_time'],
                'first_seen_round': doc_first_seen_round[docid],
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + '\n')

    print(json.dumps({
        'output_dir': args.output_dir,
        'base_round': base_round,
        'num_documents': len(ordered_docids),
        'rounds': manifest['rounds'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
