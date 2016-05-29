#!/usr/bin/env python
import os, sys, shutil
import argparse
import urllib2
import zipfile
import json
from subprocess import check_output
import datetime
import matplotlib.pyplot as plt


def fetch_url(url, target_dir):
    file = os.path.join(target_dir, url.split('/')[-1])
    u = urllib2.urlopen(url)
    f = open(file, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file, file_size)

    file_size_dl = 0
    block_sz = 16384
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. /
                                       file_size)
        status = status + chr(8) * (len(status) + 1)
        print status,
    f.close()
    return file


def unzip_file(file, target_dir):
    f = open(file, 'rb')
    print "Unzipping: %s" % file
    z = zipfile.ZipFile(f)
    for name in z.namelist():
        z.extract(name, target_dir)
    f.close()
    os.remove(file)


def evaluate_sequence(sequence, algorithm, executable, percent, img_files,
                      gt_files):
    res = []
    for i in range(len(img_files) - 1):
        sys.stdout.write("Algorithm: %-20s Sequence: %-10s Done: [%3d/%3d]\r" %
                         (algorithm, sequence, i, len(img_files) - 1)),
        sys.stdout.flush()
        if int(percent * i) != int(percent * (i + 1)):
            res_string = check_output([executable, img_files[i], img_files[i + 1],
                                       algorithm, gt_files[i]])
            res.append(parse_evaluation_result(res_string, i))
    return res


def parse_evaluation_result(input_str, i):
    res = {}
    lines = input_str.split('\n')
    res['frame_number'] = i + 1
    res['time'] = float(lines[1].split(':')[1])
    res['error'] = {}
    res['error']['average'] = float(lines[3].split(':')[1])
    res['error']['std'] = float(lines[4].split(':')[1])
    res['error']['R0.5'] = float(lines[5][:-2].split(':')[1])
    res['error']['R1.0'] = float(lines[6][:-2].split(':')[1])
    res['error']['R2.0'] = float(lines[7][:-2].split(':')[1])
    res['error']['R5.0'] = float(lines[8][:-2].split(':')[1])
    res['error']['R10.0'] = float(lines[9][:-2].split(':')[1])
    res['error']['A0.50'] = float(lines[10].split(':')[1])
    res['error']['A0.75'] = float(lines[11].split(':')[1])
    res['error']['A0.95'] = float(lines[12].split(':')[1])
    return res


#############################DATSET DEFINITIONS################################
def fetch_mpi_sintel(target_dir):
    for url in [
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip",
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip"
    ]:
        file = fetch_url(url, target_dir)
        unzip_file(file, os.path.join(target_dir, 'mpi_sintel'))


def evaluate_mpi_sintel(source_dir, algorithm, evaluation_executable, percent):
    evaluation_result = {}
    img_dir = os.path.join(source_dir, 'mpi_sintel', 'training', 'final')
    gt_dir = os.path.join(source_dir, 'mpi_sintel', 'training', 'flow')
    sequences = [f for f in os.listdir(img_dir)
                 if os.path.isdir(os.path.join(img_dir, f))]
    for seq in sequences:
        img_files = sorted([os.path.join(img_dir, seq, f)
                            for f in os.listdir(os.path.join(img_dir, seq))
                            if f.endswith(".png")])
        gt_files = sorted([os.path.join(gt_dir, seq, f)
                           for f in os.listdir(os.path.join(gt_dir, seq))
                           if f.endswith(".flo")])
        evaluation_result[seq] = evaluate_sequence(
            seq, algorithm, evaluation_executable, percent, img_files,
            gt_files)
    return evaluation_result


def fetch_middlebury(target_dir):
    for url in [
            "http://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip",
            "http://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip"
    ]:
        file = fetch_url(url, target_dir)
        unzip_file(file, os.path.join(target_dir, 'middlebury'))


def evaluate_middlebury(source_dir, algorithm, evaluation_executable, percent):
    evaluation_result = {}
    img_dir = os.path.join(source_dir, 'middlebury', 'other-data')
    gt_dir = os.path.join(source_dir, 'middlebury', 'other-gt-flow')
    sequences = [f for f in os.listdir(gt_dir)
                 if os.path.isdir(os.path.join(gt_dir, f))]
    for seq in sequences:
        img_files = sorted([os.path.join(img_dir, seq, f)
                            for f in os.listdir(os.path.join(img_dir, seq))
                            if f.endswith(".png")])
        gt_files = sorted([os.path.join(gt_dir, seq, f)
                           for f in os.listdir(os.path.join(gt_dir, seq))
                           if f.endswith(".flo")])
        evaluation_result[seq] = evaluate_sequence(
            seq, algorithm, evaluation_executable, 1.0, img_files, gt_files)
    return evaluation_result


datasets = {
    "mpi_sintel": {
        "fetch_function": fetch_mpi_sintel,
        "evaluate_function": evaluate_mpi_sintel
    },
    "middlebury": {
        "fetch_function": fetch_middlebury,
        "evaluate_function": evaluate_middlebury
    }
}

###############################################################################


def load_json(path):
    f = open(path, "r")
    data = json.load(f)
    return data


def save_json(obj, path):
    f = open(path, "w")
    json.dump(obj, f, indent=2)


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def parse_sequence(input_str):
    if len(input_str) == 0:
        return []
    else:
        return [o.strip() for o in input_str.split(",") if o]


def build_chart(dst_folder, src_folder, percent, dataset):
    fig = plt.figure(figsize=(16, 10))
    src_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder)
                 if f.endswith(".json")]
    markers = ["o", "s", "h", "^", "D"]
    marker_idx = 0
    colors = ["b", "g", "r"]
    color_idx = 0
    for file in src_files:
        data = load_json(file)
        name = os.path.basename(file).split('.')[0]
        average_time = 0.0
        average_error = 0.0
        num_elem = 0
        for seq in data.keys():
            for frame in data[seq]:
                average_time += frame["time"]
                average_error += frame["error"]["average"]
                num_elem += 1
        average_time /= num_elem
        average_error /= num_elem

        marker_style = colors[color_idx] + markers[marker_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
        marker_idx += 1
        if marker_idx >= len(markers):
            marker_idx = 0
        plt.gca().plot([average_time], [average_error],
                       marker_style,
                       markersize=14,
                       label=name)

    plt.gca().set_ylabel('Average Endpoint Error (EPE)', fontsize=20)
    plt.gca().set_xlabel('Average Runtime (seconds per frame)', fontsize=20)
    plt.gca().set_xscale("log")
    if int(percent) == 100:
        plt.gca().set_title('Evaluation on ' + dataset, fontsize=20)
    else:
        plt.gca().set_title(
            'Evaluation on ' + dataset + ' (' + percent + '% of all frames)',
            fontsize=20)
    plt.gca().legend()
    fig.savefig(
        os.path.join(dst_folder, "evaluation_results_" + dataset + "_p" +
                     percent + ".png"),
        bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optical flow benchmarking script',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "bin_path",
        default=".",
        help="""Path to the directory with built samples (should contain
                optflow-example-optical_flow_evaluation)""")
    parser.add_argument(
        "-a",
        "--algorithms",
        metavar="ALGORITHMS",
        default="",
        help="""Comma-separated list of optical-flow algorithms to evaluate
                (example: -a farneback,tvl1,deepflow). Note that previously
                evaluated algorithms are also included in the output charts""")
    parser.add_argument(
        "-d",
        "--datasets",
        metavar="DATASETS",
        default="mpi_sintel",
        help="""Comma-separated list of datasets for evaluation (currently only
                'mpi_sintel' and 'middlebury' are supported)""")
    parser.add_argument(
        "-w",
        "--cwd",
        metavar="WORKING_DIR",
        default="./OF_evaluation_data",
        help="Working directory for storing datasets and intermediate results")
    parser.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        default="./OF_evaluation_results",
        help="Output directory where to store benchmark results (comparative charts)")
    parser.add_argument(
        "-p",
        "--percent",
        metavar="PERCENT",
        default="100",
        help="""Percent of the dataset to consider (the lower this value is the faster
                the evaluation will finish; it, however, naturally decreases the
                evaluation accuracy)""")
    args, other_args = parser.parse_known_args()

    create_dir(args.cwd)
    current_state = {"fetched_datasets": []}
    if os.path.isfile(os.path.join(args.cwd, "state.json")):
        current_state = load_json(os.path.join(args.cwd, "state.json"))

    algorithm_list = parse_sequence(args.algorithms)
    dataset_list = parse_sequence(args.datasets)
    for dataset in dataset_list:
        if dataset not in datasets.keys():
            print "Error: unsupported dataset " + dataset
            sys.exit(1)
        if dataset not in current_state["fetched_datasets"]:
            datasets[dataset]["fetch_function"](args.cwd)
            current_state["fetched_datasets"].append(dataset)
            save_json(current_state, os.path.join(args.cwd, "state.json"))

    create_dir(os.path.join(args.cwd, 'evaluation_results'))
    for dataset in dataset_list:
        create_dir(os.path.join(args.cwd, 'evaluation_results', dataset))
        create_dir(os.path.join(args.cwd, 'evaluation_results', dataset, 'p' +
                                str(args.percent)))
        for algorithm in algorithm_list:
            eval_res = datasets[dataset]["evaluate_function"](
                args.cwd, algorithm, os.path.join(
                    args.bin_path, 'optflow-example-optical_flow_evaluation'),
                float(args.percent) / 100)
            save_json(eval_res, os.path.join(
                args.cwd, 'evaluation_results', dataset, 'p' + str(args.percent),
                algorithm + datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M") +
                ".json"))

    create_dir(args.out)
    for dataset in dataset_list:
        build_chart(args.out, os.path.join(args.cwd, 'evaluation_results',
                                           dataset, 'p' + str(args.percent)),
                    args.percent, dataset)
