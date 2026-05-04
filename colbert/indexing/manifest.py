import os
import ujson


MANIFEST_FILENAME = 'manifest.json'


def manifest_path(index_path):
    return os.path.join(index_path, MANIFEST_FILENAME)


def load_manifest(index_path):
    path = manifest_path(index_path)
    if not os.path.exists(path):
        return None

    with open(path, mode='r', encoding='utf-8') as input_file:
        return ujson.load(input_file)


def save_manifest(index_path, manifest):
    path = manifest_path(index_path)
    with open(path, mode='w', encoding='utf-8') as output_file:
        ujson.dump(manifest, output_file, indent=2)
        output_file.write('\n')


def upsert_segment(index_path, name, relative_path, segment_type='base', faiss_name=None, active=True):
    manifest = load_manifest(index_path) or {'version': 1, 'segments': []}
    segments = manifest.setdefault('segments', [])

    segment = next((item for item in segments if item['name'] == name), None)
    if segment is None:
        segment = {'name': name}
        segments.append(segment)

    segment['relative_path'] = relative_path
    segment['segment_type'] = segment_type
    segment['active'] = active
    if faiss_name is not None:
        segment['faiss_name'] = faiss_name
    else:
        segment.setdefault('faiss_name', None)

    save_manifest(index_path, manifest)
    return manifest


def initialize_base_manifest(index_path):
    return upsert_segment(index_path, name='base', relative_path='.', segment_type='base', faiss_name=None, active=True)


def deactivate_segment(index_path, name):
    manifest = load_manifest(index_path)
    if manifest is None:
        return None

    for segment in manifest.get('segments', []):
        if segment['name'] == name:
            segment['active'] = False
            save_manifest(index_path, manifest)
            return manifest

    return manifest


def list_active_segments(index_path, default_faiss_name=None):
    manifest = load_manifest(index_path)
    if manifest is None or len(manifest.get('segments', [])) == 0:
        return [{
            'name': 'base',
            'relative_path': '.',
            'segment_type': 'base',
            'faiss_name': default_faiss_name,
            'active': True,
        }]

    segments = []
    for segment in manifest.get('segments', []):
        if not segment.get('active', True):
            continue

        item = dict(segment)
        if item.get('faiss_name') is None and item.get('relative_path', '.') == '.':
            item['faiss_name'] = default_faiss_name
        segments.append(item)

    return segments
