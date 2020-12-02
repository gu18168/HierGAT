from argparse import ArgumentParser
from pathlib import Path
import pickle

from data import process, write_pickle

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        action='append', dest='input_files',
                        help='Give input csv file to process')
    parser.add_argument('-d', '--dir', required=True,
                        dest='input_dir',
                        help="Give input file's dir to process")
    parser.add_argument('-o', '--output', required=True,
                        dest='output_dir',
                        help="Give output dir to store")
    parser.add_argument('-m', '--mapping', required=True,
                        dest='map_file',
                        help='Give mapping file to evaluation')
    parser.add_argument('-mo', '--model', required=True,
                        dest='model_file',
                        help='Give the model to process node')
    parser.add_argument('-l', '--limit', type=int, default=16,
                        dest='limit',
                        help='The number of negative sample')
    parser.add_argument('-p', '--pattern', required=True,
                        dest='pattern',
                        help='Give the pattern for resolve the input to entity')
    parser.add_argument('-mp', '--map-pattern', default='l_id,r_id',
                        dest='map_pattern',
                        help='Give the pattern for resolve the mapping')
    parser.add_argument('-id', default='id',
                        dest='id',
                        help='Give the id of entity')

    args = parser.parse_args()

    attrs_adjs, attrs_lines, attrs_features, labels = \
        process(args.input_files, args.input_dir, args.map_file,
                args.limit, args.pattern, args.map_pattern, args.id, args.model_file)

    # Store in the file
    write_pickle(Path(args.output_dir, 'adjs'), attrs_adjs)
    write_pickle(Path(args.output_dir, 'lines'), attrs_lines)
    write_pickle(Path(args.output_dir, 'features'), attrs_features)
    write_pickle(Path(args.output_dir, 'labels'), labels)
