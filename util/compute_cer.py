import sys
import argparse
import json
import matplotlib.pyplot as plt
from jiwer import cer, wer
import editdistance


def espnet_wer(seqs_true, seqs_hat):
    """Calculate sentence-level WER score.

    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level WER score
    :rtype float
    """
    if isinstance(seqs_true, str) and isinstance(seqs_hat, str):
        seqs_true = [seqs_true]
        seqs_hat = [seqs_hat]
    
    word_eds, word_ref_lens = [], []
    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)


def compute_error_rate(metric_type, ground_truth, hypothesis):
    if metric_type == "cer":
        result = cer(ground_truth, hypothesis)
    elif metric_type == "wer":
        result = espnet_wer(ground_truth, hypothesis)
    return result


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps", type=str, required=True, 
        help="input hyps text json file")
    parser.add_argument("--ref", type=str, required=True, 
        help="input ref text json file")
    parser.add_argument("--n_best", type=str, required=True, 
        help="n best")
    parser.add_argument("--save_path", type=str, required=False)
    parser.add_argument("--metric_type", type=str, required=True)
    args = parser.parse_args()

    hyps_json = json.load(
        open(args.hyps, "r", encoding="utf-8")
    )
    ref_json = json.load(
        open(args.ref, "r", encoding="utf-8")
    )

    n_best = int(args.n_best)
    oracle_hyps, top_one_list = [], []
    oracle_distribution = {pos: 0 for pos in range(n_best)}
    for (utt_id, ref), hyps in zip(ref_json.items(), hyps_json.values()):
        min_er = sys.maxsize

        if isinstance(hyps, dict):
            hyps = list(hyps.values())[:n_best]
        elif isinstance(hyps, str):
            hyps = [hyps]

        top_one_list.append(hyps[0])

        for pos, hyp in enumerate(hyps):
            error_rate = compute_error_rate(args.metric_type, ref, hyp)
            if error_rate < min_er:
                min_er = error_rate
                oracle_pos = pos
                oracle_hyp = hyp
        
        oracle_distribution[oracle_pos] += 1
        oracle_hyps.append(oracle_hyp)

    ref_list = list(ref_json.values())
    oracle_er = compute_error_rate(args.metric_type, ref_list, oracle_hyps)
    top_one_er = compute_error_rate(args.metric_type, ref_list, top_one_list)

    # show result
    print(f"top-1 er: {top_one_er}")
    print(f"oracle er: {oracle_er}")
    print("oracle distribution: {pos in n-best: count} => ", oracle_distribution)
    plot = plt.bar(
        oracle_distribution.keys(),
        oracle_distribution.values(),
    )
    plt.xlabel("oracle position in n-best")
    plt.ylabel("oracle count")
    plt.show()
    if args.save_path != None:
        plt.savefig(args.save_path)