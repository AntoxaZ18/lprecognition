import numpy as np
from time import time

class IdGen:
    def __init__(self, thread_id, max_count = 1024):
        self.thread_id = thread_id
        self.max_count = max_count
        self.counter = 0

    def __call__(self):
        now = int(time() * 1e6)  # Микросекунды
        uuid = f"{now}-{self.thread_id}-{self.counter}"
        self.counter += 1
        return uuid

CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

class Decoder:
    """Interface for sequence decoding"""
    def decode(self, predicted_seq, chars_list):
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def decode(self, predicted_seq, chars_list):
        full_pred_labels = []
        labels = []
        # predicted_seq.shape = [batch, len(chars_list), len_seq]
        for i in range(predicted_seq.shape[0]):
            single_prediction = predicted_seq[i, :, :]
            predicted_labels = []
            for j in range(single_prediction.shape[1]):
                predicted_labels.append(np.argmax(single_prediction[:, j], axis=0))

            without_repeating = []
            current_char = predicted_labels[0]
            if current_char != len(chars_list) - 1:
                without_repeating.append(current_char)
            for c in predicted_labels:
                if (current_char == c) or (c == len(chars_list) - 1):
                    if c == len(chars_list) - 1:
                        current_char = c
                    continue
                without_repeating.append(c)
                current_char = c

            full_pred_labels.append(without_repeating)

        for i, label in enumerate(full_pred_labels):
            decoded_label = ''
            for j in label:
                decoded_label += chars_list[j]
            labels.append(decoded_label)

        return labels, full_pred_labels


class BeamDecoder(Decoder):
    def decode(self, predicted_seq, chars_list):

        labels = []
        final_labels = []
        final_prob = []
        k = 1
        for i in range(predicted_seq.shape[0]):
            sequences = [[list(), 0.0]]
            all_seq = []
            single_prediction = predicted_seq[i, :, :]
            for j in range(single_prediction.shape[1]):
                single_seq = []
                for char in single_prediction[:, j]:
                    single_seq.append(char)
                all_seq.append(single_seq)

            for row in all_seq:
                all_candidates = []
                for i in range(len(sequences)):
                    seq, score = sequences[i]
                    for j in range(len(row)):
                        candidate = [seq + [j], score - row[j]]

                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                # select k best
                sequences = ordered[:k]

            full_pred_labels = []
            probs = []
            for i in sequences:

                predicted_labels = i[0]
                without_repeating = []
                current_char = predicted_labels[0]
                if current_char != len(chars_list) - 1:
                    without_repeating.append(current_char)
                for c in predicted_labels:
                    if (current_char == c) or (c == len(chars_list) - 1):
                        if c == len(chars_list) - 1:
                            current_char = c
                        continue
                    without_repeating.append(c)
                    current_char = c

                full_pred_labels.append(without_repeating)
                probs.append(i[1])
            for i, label in enumerate(full_pred_labels):
                decoded_label = ''
                for j in label:
                    decoded_label += chars_list[j]
                labels.append(decoded_label)
                final_prob.append(probs[i])
                final_labels.append(full_pred_labels[i])

        return labels, final_prob, final_labels


def decode_function(predicted_seq, chars_list, decoder=GreedyDecoder):
    return decoder().decode(predicted_seq, chars_list)
