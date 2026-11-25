"""Document QA with a supplied context; this is not a retrieval/RAG system."""

from pathlib import Path
import argparse
import json

import torch
from qa_model import (CHECKPOINT_DIR, DEVICE, CONFIG, get_model, get_tokenizer,
                      predict_examples, seed_everything)


def load_document_model():
    selection_path = CHECKPOINT_DIR / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError('Run qa_finetuning.ipynb to train/select a local QA checkpoint first.')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    assert selection['config'] == CONFIG, 'Checkpoint configuration mismatch'
    length = selection['max_length']
    seed_everything()
    tokenizer = get_tokenizer()
    model = get_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / str(length) / 'best.pt',
                                    map_location=DEVICE, weights_only=True))
    model.eval()
    return model, tokenizer, length


def answer_question(question, context, model, tokenizer, max_length):
    if not question.strip() or not context.strip():
        raise ValueError('Question and context must be non-empty.')
    row = dict(id='document', title='supplied document', question=question,
               context=context, answers=[])
    answer = predict_examples(model, [row], tokenizer, max_length)[0]
    return dict(answer=answer['text'], start_char=answer['start_char'],
                end_char=answer['end_char'], score=answer['score'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--context')
    source.add_argument('--file', type=Path)
    args = parser.parse_args()
    context = args.context if args.context is not None else args.file.read_text(encoding='utf-8')
    model, tokenizer, length = load_document_model()
    print(json.dumps(answer_question(args.question, context, model, tokenizer, length),
                     ensure_ascii=False, indent=2))
