import collections
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("miglss/mdeberta-v3-base-konturDS")

def find_labels(offsets, answer_start, answer_end, sequence_ids):
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # If both answer_start and answer_end are 0, return (0, 0) to represent "no answer"
    if answer_start == 0 and answer_end == 0:
        return 0, 0

    # If the answer is not fully in the context or answer_start is greater than answer_end, return (-1, -1)
    if (
        offsets[context_start][0] > answer_end
        or offsets[context_end][1] < answer_start
        or answer_start > answer_end
    ):
        return -1, -1

    idx = context_start
    while idx <= context_end and offsets[idx][0] <= answer_start:
        idx += 1
    start_position = idx - 1

    idx = context_end
    while idx >= context_start and offsets[idx][1] >= answer_end:
        idx -= 1
    end_position = idx + 1

    return start_position, end_position

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=256,
        stride=186,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["start_positions"] = []
    inputs["end_positions"] = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        start, end = find_labels(
            offset, examples["answer_start"][sample_idx], examples["answer_end"][sample_idx], inputs.sequence_ids(i)
        )
        
        inputs["start_positions"].append(start)
        inputs["end_positions"].append(end)

    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=256,
        stride=186,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    
    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["start_positions"] = []
    inputs["end_positions"] = []
    inputs["example_id"] = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        inputs["example_id"].append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset_mapping[i] = [(o if s == 1 else None) for o, s in zip(offset, sequence_ids)]
        start, end = find_labels(
            offset, examples["answer_start"][sample_idx], examples["answer_end"][sample_idx], sequence_ids
        )
        
        inputs["start_positions"].append(start)
        inputs["end_positions"].append(end)

    return inputs

def postprocess_predictions(
    examples,
    features,
    raw_predictions, 
    n_best_size = 20,
    max_answer_length = 100):
  

    all_start_logits, all_end_logits = raw_predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(tqdm(examples, leave=True)):
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions[example["id"]] = answer

    return predictions