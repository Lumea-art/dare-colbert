import os
import math
import random
import ujson

from colbert.utils.runs import Run
from colbert.utils.parser import Arguments
import colbert.utils.distributed as distributed

from colbert.utils.utils import print_message, create_directory, timestamp
from colbert.indexing.encoder import CollectionEncoder
from colbert.indexing.faiss import index_faiss, get_faiss_index_name
from colbert.indexing.loaders import load_doclens
from colbert.indexing.manifest import initialize_base_manifest, upsert_segment


def main():
    random.seed(12345)

    parser = Arguments(description='Append a delta segment for incremental ColBERT retrieval.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()

    parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)
    parser.add_argument('--partitions', dest='partitions', default=None, type=int)
    parser.add_argument('--sample', dest='sample', default=None, type=float)
    parser.add_argument('--slices', dest='slices', default=1, type=int)
    parser.add_argument('--delta-name', dest='delta_name', default=None, type=str)

    args = parser.parse()
    assert args.slices == 1, "Incremental retrieval currently requires --slices 1."

    with Run.context():
        base_index_path = os.path.join(args.index_root, args.index_name)
        assert os.path.exists(base_index_path), base_index_path
        initialize_base_manifest(base_index_path)

        delta_name = args.delta_name or f"seg_{timestamp().replace('-', '').replace(':', '').replace('.', '').replace('_', '')}"
        delta_root = os.path.join(base_index_path, 'delta')
        args.index_path = os.path.join(delta_root, delta_name)
        assert not os.path.exists(args.index_path), args.index_path

        distributed.barrier(args.rank)

        if args.rank < 1:
            create_directory(delta_root)
            create_directory(args.index_path)

        distributed.barrier(args.rank)

        process_idx = max(0, args.rank)
        encoder = CollectionEncoder(args, process_idx=process_idx, num_processes=args.nranks)
        encoder.encode()

        distributed.barrier(args.rank)

        if args.rank < 1:
            metadata_path = os.path.join(args.index_path, 'metadata.json')
            segment_metadata = dict(args.input_arguments.__dict__)
            segment_metadata['segment_type'] = 'delta'
            segment_metadata['delta_name'] = delta_name
            print_message("Saving (the following) metadata to", metadata_path, "..")
            print(segment_metadata)

            with open(metadata_path, 'w') as output_metadata:
                ujson.dump(segment_metadata, output_metadata)

            num_embeddings = sum(load_doclens(args.index_path))
            print("#> num_embeddings =", num_embeddings)

            if args.partitions is None:
                args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
                Run.warn("You did not specify --partitions!")
                Run.warn("Default computation chooses", args.partitions,
                         "partitions (for {} embeddings)".format(num_embeddings))

            index_faiss(args)
            faiss_name = get_faiss_index_name(args)

            upsert_segment(
                base_index_path,
                name=delta_name,
                relative_path=os.path.relpath(args.index_path, base_index_path),
                segment_type='delta',
                faiss_name=faiss_name,
            )

        distributed.barrier(args.rank)


if __name__ == "__main__":
    main()
