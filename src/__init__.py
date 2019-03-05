import os
from pathlib import Path
import sys
from six.moves import urllib
import tensorflow as tf
import tarfile
import numpy
import re

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
MODEL_DIR = 'model/'


class NodeLookup(object):

    def __init__(self, label_path=None, id_path=None):
        if not label_path:
            label_path = os.path.join(
                MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt'
            )
        if not id_path:
            id_path = os.path.join(
                MODEL_DIR, 'imagenet_synset_to_human_label_map.txt'
            )
        self.node_lookup = self.load(label_path, id_path)

    def load(self, label_path, id_path):
        if not tf.gfile.Exists(id_path):
            tf.logging.fatal('File does not exist %s', id_path)
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_path)

        proto_lines = tf.gfile.GFile(id_path).readlines()
        human_readable_content = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_lines:
            parsed_items = p.findall(line)
            _id = parsed_items[0]
            human_string = parsed_items[2]
            human_readable_content[_id] = human_string

        node_id_to_uid = {}
        proto = tf.gfile.GFile(label_path).readlines()
        for line in proto:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in human_readable_content:
                tf.logging.fatal('Failed to locate: %s', val)
            name = human_readable_content[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def main(_):
    download_model()
    selection = input("enter a path to format:")
    print(selection)
    print(os.path.exists(selection))
    if os.path.exists(selection):
        crawl_dir(selection)
    else:
        print("Invalid path")


def crawl_dir(selection):
    jpg_list = Path(selection).glob('*.jpg')
    png_list = Path(selection).glob('*.png')

    for path in jpg_list:
        print(path)
        detect_image_content(selection, str(path))
    for path in png_list:
        detect_image_content(selection, str(path))


def download_model():
    dest_dir = MODEL_DIR
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_dir)


def detect(file):
    if not tf.gfile.Exists(file):
        tf.logging.fatal('File does not exist %s', file)
    data = tf.gfile.FastGFile(file, 'rb').read()

    with tf.gfile.FastGFile(os.path.join( MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name('softmax:0')
        predictions = session.run(softmax_tensor, { 'DecodeJpeg/contents:0': data})
        predictions = numpy.squeeze(predictions)

        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-5:][::-1]
        human_strings = [node_lookup.id_to_string(top_k[0]), node_lookup.id_to_string(top_k[1]), node_lookup.id_to_string(top_k[2]), node_lookup.id_to_string(top_k[3]), node_lookup.id_to_string(top_k[4])]
        scores = [predictions[top_k[0]], predictions[top_k[1]], predictions[top_k[2]], predictions[top_k[3]], predictions[top_k[4]]]

    return [human_strings, scores]


def detect_image_content(dir, filepath):
    print(filepath)
    result = detect(filepath)
    print(result)
    print(normalize_string_for_filename(result[0][0]))
    _, file_extension = os.path.splitext(filepath)
    path = os.path.join(dir, normalize_string_for_filename(result[0][0]) + file_extension)
    print(path)

    try:
        os.rename(filepath, path)
    except FileExistsError:
        os.rename(filepath, os.path.join(dir, normalize_string_for_filename(result[0][1]) + file_extension))


def normalize_string_for_filename(string):
    return string.replace(" ", "_").replace(",", "_")


if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])
