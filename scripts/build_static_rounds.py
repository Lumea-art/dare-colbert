import argparse
import json
import os
import shutil
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build full static round-by-round evaluation directories from a base+delta incremental dataset.'
    )
    parser.add_argument('--incremental-root', dest='incremental_root', required=True,
                        help='Processed incremental dataset root, e.g. data/processed/trec_covid')
    parser.add_argument('--output-dir', dest='output_dir', required=True,
                        help='Directory for static round outputs, e.g. data/processed/trec_covid_static')
    parser.add_argument('--rounds', dest='rounds', nargs='+', type=int, default=None,
                        help='Subset of rounds to build. Defaults to all rounds in manifest.json.')
    parser.add_argument('--skip-meta', dest='skip_meta', default=False, action='store_true',
                        help='Do not build collection_meta.jsonl even if source metadata is available.')
    return parser.parse_args()


def load_json(path):
    with open(path, mode='r', encoding='utf-8') as input_file:
        return json.load(input_file)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def build_segment_plan(root, manifest, target_round):
    base_round = int(manifest['base_round'])
    rounds = ordered_round_ids(manifest)
    assert target_round in rounds, (target_round, rounds)

    collection_paths = [os.path.join(root, 'base', 'collection.tsv')]
    meta_paths = [os.path.join(root, 'base', 'collection_meta.jsonl')]
    segment_names = ['base']

    for round_id in rounds:
        if round_id > target_round:
            break
        if round_id == base_round:
            continue

        round_info = manifest['rounds'][str(round_id)]
        delta_collection = round_info.get('delta_collection_path')
        delta_meta = round_info.get('delta_meta_path')

        if delta_collection is not None:
            collection_paths.append(resolve_relative(root, delta_collection))
            segment_names.append(f'round{round_id}')
        if delta_meta is not None:
            meta_paths.append(resolve_relative(root, delta_meta))

    return {
        'collection_paths': collection_paths,
        'meta_paths': meta_paths,
        'segment_names': segment_names,
    }


def parse_pid_from_tsv(line, path, line_number):
    parts = line.split('\t', 1)
    if len(parts) < 2:
        raise ValueError(f'Malformed TSV line in {path}:{line_number}')
    return parts[0]


def concat_tsv_files(input_paths, output_path):
    total = 0
    seen = set()

    with open(output_path, mode='w', encoding='utf-8') as output_file:
        for input_path in input_paths:
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path)

            with open(input_path, mode='r', encoding='utf-8') as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    line = line.rstrip('\n')
                    if not line:
                        continue

                    pid = parse_pid_from_tsv(line, input_path, line_number)
                    if pid in seen:
                        # [修改点1] 遇到重复的 pid 直接跳过，不再抛出异常
                        continue

                    seen.add(pid)
                    output_file.write(line + '\n')
                    total += 1

    return total


def concat_meta_files(input_paths, output_path):
    total = 0
    seen = set()

    with open(output_path, mode='w', encoding='utf-8') as output_file:
        for input_path in input_paths:
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path)

            with open(input_path, mode='r', encoding='utf-8') as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    line = line.rstrip('\n')
                    if not line:
                        continue

                    payload = json.loads(line)
                    pid = str(payload['pid'])
                    if pid in seen:
                        # [修改点2] 跳过重复的元数据
                        continue

                    seen.add(pid)
                    output_file.write(json.dumps(payload, ensure_ascii=False) + '\n')
                    total += 1

    return total


def source_round_dir(root, round_id):
    return os.path.join(root, 'rounds', f'round{round_id}')


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        return True
    return False


def main():
    args = parse_args()

    manifest_path = os.path.join(args.incremental_root, 'manifest.json')
    manifest = load_json(manifest_path)
    rounds = select_rounds(manifest, args.rounds)

    ensure_dir(args.output_dir)

    output_manifest = {
        'dataset': manifest.get('dataset', 'static-rounds'),
        'source_incremental_root': os.path.abspath(args.incremental_root),
        'rounds': OrderedDict(),
    }

    for round_id in rounds:
        round_info = manifest['rounds'][str(round_id)]
        round_output_dir = os.path.join(args.output_dir, f'round{round_id}')
        ensure_dir(round_output_dir)

        segment_plan = build_segment_plan(args.incremental_root, manifest, round_id)
        output_collection = os.path.join(round_output_dir, 'collection.tsv')
        num_docs = concat_tsv_files(segment_plan['collection_paths'], output_collection)

        expected_docs = round_info.get('num_cumulative_docs')
        # [修改点3] 因为去重了，数量肯定对不上，把报错改为仅打印警告
        if expected_docs is not None and num_docs != expected_docs:
            print(f'[WARN] Round {round_id}: Deduplicated collection has {num_docs} docs, manifest expected {expected_docs}.')

        meta_written = False
        output_meta = os.path.join(round_output_dir, 'collection_meta.jsonl')
        if not args.skip_meta and all(os.path.exists(path) for path in segment_plan['meta_paths']):
            meta_docs = concat_meta_files(segment_plan['meta_paths'], output_meta)
            # [修改点4] 同理，把元数据数量校验的异常也改成警告
            if meta_docs != num_docs:
                print(f'[WARN] Round {round_id} has {meta_docs} metadata rows but {num_docs} collection rows.')
            meta_written = True

        queries_src = resolve_relative(args.incremental_root, round_info['queries_path'])
        qrels_src = resolve_relative(args.incremental_root, round_info['qrels_path'])
        queries_dst = os.path.join(round_output_dir, 'queries.tsv')
        qrels_dst = os.path.join(round_output_dir, 'qrels.tsv')

        shutil.copyfile(queries_src, queries_dst)
        shutil.copyfile(qrels_src, qrels_dst)

        cumulative_docids_src = os.path.join(source_round_dir(args.incremental_root, round_id), 'cumulative_docids.txt')
        cumulative_docids_dst = os.path.join(round_output_dir, 'cumulative_docids.txt')
        copy_if_exists(cumulative_docids_src, cumulative_docids_dst)

        output_manifest['rounds'][str(round_id)] = {
            'num_documents': num_docs,
            'segments': segment_plan['segment_names'],
            'collection_path': os.path.relpath(output_collection, args.output_dir),
            'collection_meta_path': None if not meta_written else os.path.relpath(output_meta, args.output_dir),
            'queries_path': os.path.relpath(queries_dst, args.output_dir),
            'qrels_path': os.path.relpath(qrels_dst, args.output_dir),
        }

    output_manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(output_manifest_path, mode='w', encoding='utf-8') as output_file:
        json.dump(output_manifest, output_file, indent=2)
        output_file.write('\n')

    print(json.dumps({
        'output_dir': args.output_dir,
        'rounds': output_manifest['rounds'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()