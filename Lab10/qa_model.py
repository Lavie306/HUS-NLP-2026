"""SQuAD data, heuristic baseline, span alignment and ALBERT extractive QA."""

from collections import Counter
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import os
import platform
import random
import re
import string
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader

LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parent
DATA_DIR = ROOT / 'models' / 'datasets' / 'lab10'
CHECKPOINT_DIR = ROOT / 'checkpoints' / 'lab10'
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, get_linear_schedule_with_warmup

MODEL_ID = 'albert/albert-base-v2'
REVISION = '8e2f239c5f8a2c0f253781ca60135db913e5c80c'
SEED = 42
CONFIG = dict(model=MODEL_ID, revision=REVISION, seed=SEED, batch_size=4,
              epochs=2, learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1,
              stride=32, max_question_tokens=64, max_answer_tokens=64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.temporary')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def atomic_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.temporary')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_source(split):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f'{split}-v1.1.json'
    if not path.exists():
        with urllib.request.urlopen(f'https://rajpurkar.github.io/SQuAD-explorer/dataset/{path.name}', timeout=120) as response:
            payload = response.read()
        json.loads(payload)
        path.write_bytes(payload)
    raw = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                row = dict(id=qa['id'], title=article['title'], context=paragraph['context'],
                           question=qa['question'], answers=qa['answers'])
                assert row['answers'] and all(row['context'][a['answer_start']:a['answer_start'] + len(a['text'])] == a['text'] for a in row['answers'])
                rows.append(row)
    assert len(rows) == (87599 if split == 'train' else 10570)
    return rows


def load_splits(train_size=1000, validation_size=200, test_size=200):
    """Official train + article-disjoint halves of official dev (no public test gold)."""
    train, dev = load_source('train'), load_source('dev')
    titles = sorted({r['title'] for r in dev})
    random.Random(SEED).shuffle(titles)
    validation_titles = set(titles[:len(titles) // 2])
    pools = dict(train=train, validation=[r for r in dev if r['title'] in validation_titles],
                 test=[r for r in dev if r['title'] not in validation_titles])
    sizes = dict(train=train_size, validation=validation_size, test=test_size)
    splits = {}
    for offset, (name, rows) in enumerate(pools.items()):
        requested = sizes[name]
        assert requested is None or 0 < requested <= len(rows)
        splits[name] = list(rows) if requested is None else random.Random(SEED + offset).sample(rows, requested)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ({r['id'] for r in splits[a]} & {r['id'] for r in splits[b]})
        assert not ({r['title'] for r in splits[a]} & {r['title'] for r in splits[b]})
    return splits


def split_signature(splits):
    return dict(source_sha256={name: hashlib.sha256((DATA_DIR / f'{name}-v1.1.json').read_bytes()).hexdigest()
                               for name in ['train', 'dev']},
                selected_ids={name: [row['id'] for row in rows] for name, rows in splits.items()})


def normalize_answer(text):
    """SQuAD normalization: lowercase, ASCII punctuation, English articles, spaces."""
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answer(prediction, gold):
    """Per-example EM/F1 on 0–1; evaluate_answers reports their means on 0–100."""
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    exact = int(p == g)
    if not p or not g:
        return dict(em=exact, f1=0.0)
    overlap = sum((Counter(p) & Counter(g)).values())
    return dict(em=exact, f1=2 * overlap / (len(p) + len(g)))


def example_score(prediction, row):
    scores = [score_answer(prediction, a['text']) for a in row['answers']]
    return {key: max(s[key] for s in scores) for key in ['em', 'f1']}


def evaluate_answers(rows, predictions):
    assert len(rows) == len(predictions) and len(rows) > 0
    scores = [example_score(p['text'], row) for row, p in zip(rows, predictions)]
    return {key: 100 * sum(s[key] for s in scores) / len(scores) for key in ['em', 'f1']}


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)


def get_model(pretrained=True):
    if pretrained:
        return AutoModelForQuestionAnswering.from_pretrained(MODEL_ID, revision=REVISION)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    return AutoModelForQuestionAnswering.from_config(config)


def prepare_question(question, tokenizer):
    question = question.strip()
    encoded = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded['input_ids']) <= CONFIG['max_question_tokens']:
        return question
    question = question[:encoded['offset_mapping'][CONFIG['max_question_tokens'] - 1][1]]
    while len(tokenizer(question, add_special_tokens=False)['input_ids']) > CONFIG['max_question_tokens']:
        question = question[:-1]
    return question


def align_span(offsets, sequence_ids, answer_start, answer_end, cls_index):
    """Character [start,end) -> inclusive context-token positions; CLS if not covered.

    Use covering tokens even when a gold boundary is inside a subword. Ignore
    question/special token offsets even if their character indices coincide.
    """
    context = [i for i, (s, off) in enumerate(zip(sequence_ids, offsets))
               if s == 1 and off[1] > off[0]]
    if not context or offsets[context[0]][0] > answer_start or offsets[context[-1]][1] < answer_end:
        return cls_index, cls_index
    starts = [i for i in context if offsets[i][0] <= answer_start < offsets[i][1]]
    ends = [i for i in context if offsets[i][0] < answer_end <= offsets[i][1]]
    if not starts or not ends or starts[0] > ends[-1]:
        return cls_index, cls_index
    return starts[0], ends[-1]


def encode_examples(rows, tokenizer, max_length, stride=None):
    stride = CONFIG['stride'] if stride is None else stride
    features, metadata = [], []
    for first in range(0, len(rows), 64):
        batch = rows[first:first + 64]
        questions = [prepare_question(r['question'], tokenizer) for r in batch]
        encoded = tokenizer(questions, [r['context'] for r in batch], truncation='only_second',
                            max_length=max_length, stride=stride, return_overflowing_tokens=True,
                            return_offsets_mapping=True, padding=False)
        for feature_id, sample_id in enumerate(encoded['overflow_to_sample_mapping']):
            row = batch[sample_id]
            offsets = encoded['offset_mapping'][feature_id]
            sequence_ids = encoded.sequence_ids(feature_id)
            ids = encoded['input_ids'][feature_id]
            cls = ids.index(tokenizer.cls_token_id)
            answer = row['answers'][0] if row['answers'] else None
            start, end = ((cls, cls) if answer is None else align_span(
                offsets, sequence_ids, answer['answer_start'], answer['answer_start'] + len(answer['text']), cls))
            item = {key: encoded[key][feature_id] for key in tokenizer.model_input_names if key in encoded}
            features.append(dict(item, start_positions=start, end_positions=end))
            metadata.append(dict(example_index=first + sample_id, example_id=row['id'],
                offsets=[list(off) if sequence_ids[i] == 1 and off[1] > off[0] else None for i, off in enumerate(offsets)],
                cls_index=cls, has_answer=start != cls))
    assert all(len(f['input_ids']) <= max_length for f in features)
    return features, metadata


def collate_features(batch, tokenizer):
    inputs = [{k: v for k, v in item.items() if k not in ['start_positions', 'end_positions']} for item in batch]
    padded = tokenizer.pad(inputs, padding=True, return_tensors='pt')
    padded['start_positions'] = torch.tensor([item['start_positions'] for item in batch])
    padded['end_positions'] = torch.tensor([item['end_positions'] for item in batch])
    return padded


def batches_for_epoch(features, epoch):
    rng = random.Random(SEED + epoch)
    indices = list(range(len(features)))
    rng.shuffle(indices)
    ordered, size = [], CONFIG['batch_size']
    for i in range(0, len(indices), size * 50):
        ordered.extend(sorted(indices[i:i + size * 50], key=lambda j: len(features[j]['input_ids'])))
    batches = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    rng.shuffle(batches)
    return batches


def make_loader(features, tokenizer, batches=None):
    from functools import partial
    kwargs = dict(collate_fn=partial(collate_features, tokenizer=tokenizer), num_workers=0,
                  generator=torch.Generator().manual_seed(SEED))
    if batches is None:
        return DataLoader(features, batch_size=CONFIG['batch_size'], **kwargs)
    return DataLoader(features, batch_sampler=batches, **kwargs)


def infer_encoded(model, rows, encoded, tokenizer):
    features, metadata = encoded
    best = [None] * len(rows)
    cursor, total_loss, feature_count = 0, 0.0, 0
    model.eval()
    with torch.inference_mode():
        for batch in make_loader(features, tokenizer):
            size = len(batch['input_ids'])
            output = model(**{k: v.to(DEVICE) for k, v in batch.items()})
            total_loss += output.loss.item() * size
            feature_count += size
            for local in range(size):
                meta = metadata[cursor]
                valid = [i for i, off in enumerate(meta['offsets']) if off is not None]
                start_logits = output.start_logits[local].cpu().numpy()
                end_logits = output.end_logits[local].cpu().numpy()
                starts = sorted(valid, key=lambda i: -start_logits[i])[:20]
                ends = sorted(valid, key=lambda i: -end_logits[i])[:20]
                for start in starts:
                    for end in ends:
                        if start > end or end - start + 1 > CONFIG['max_answer_tokens']:
                            continue
                        char_start, char_end = meta['offsets'][start][0], meta['offsets'][end][1]
                        if char_end <= char_start:
                            continue
                        score = float(start_logits[start] + end_logits[end])
                        index = meta['example_index']
                        if best[index] is None or score > best[index]['score']:
                            best[index] = dict(text=rows[index]['context'][char_start:char_end],
                                start_char=char_start, end_char=char_end, score=score,
                                feature_index=cursor, start_token=start, end_token=end)
                cursor += 1
    for i, result in enumerate(best):
        if result is None:
            best[i] = dict(text='', start_char=0, end_char=0, score=None,
                           feature_index=None, start_token=None, end_token=None)
        assert rows[i]['context'][best[i]['start_char']:best[i]['end_char']] == best[i]['text']
    assert cursor == len(features)
    return best, total_loss / feature_count if feature_count else 0.0


def predict_examples(model, rows, tokenizer, max_length):
    return infer_encoded(model, rows, encode_examples(rows, tokenizer, max_length), tokenizer)[0]


STOPWORDS = set('a an the is are was were be been being of to in on at for from by with and or as that this what which who when where why how does did do has have had its it'.split())


def word_terms(text):
    return [w for w in re.findall(r'\w+', text.lower()) if w not in STOPWORDS]


def sentence_ranges(context):
    boundaries = [0] + [m.end() for m in re.finditer(r'[.!?]\s+', context)] + [len(context)]
    return [(a, b) for a, b in zip(boundaries, boundaries[1:]) if context[a:b].strip()]


class KeywordBaseline:
    """Select a sentence by lexical overlap, then a simple answer-shaped span."""
    def fit(self, rows):
        contexts = sorted({row['context'] for row in rows})
        counts = Counter(term for text in contexts for term in set(word_terms(text)))
        self.idf = {term: math.log((len(contexts) + 1) / (count + 1)) + 1 for term, count in counts.items()}
        return self

    def predict_one(self, row):
        context, question = row['context'], row['question']
        terms = set(word_terms(question))
        ranges = sentence_ranges(context)
        if not ranges:
            return dict(text='', start_char=0, end_char=0)
        def overlap(pair):
            a, b = pair
            return sum(self.idf.get(term, 1) for term in sorted(terms & set(word_terms(context[a:b]))))
        a, b = max(ranges, key=overlap)
        text = context[a:b]
        tokens = list(re.finditer(r'\b[\w]+(?:[\x27’-][\w]+)*\b', text))
        candidates = []
        for i in range(len(tokens)):
            for size in range(1, min(6, len(tokens) - i) + 1):
                start, end = tokens[i].start(), tokens[i + size - 1].end()
                words = [m.group() for m in tokens[i:i + size]]
                useful = [w.lower() for w in words if w.lower() not in STOPWORDS]
                if not useful:
                    continue
                score = sum(w not in terms for w in useful) / len(useful) - 0.18 * (size - 1)
                score -= sum(w in terms for w in useful)
                lowq = question.lower()
                if lowq.startswith(('when', 'how many', 'how much', 'what year', 'in what year')):
                    score += 3 * any(any(c.isdigit() for c in w) for w in words)
                elif lowq.startswith(('who', 'where', 'which country', 'what city')):
                    score += 1.5 * all(w[0].isupper() or w.lower() in ['of', 'the', 'and'] for w in words)
                candidates.append((score, -start, start, end))
        if not candidates:
            return dict(text=text.strip(), start_char=a + len(text) - len(text.lstrip()),
                        end_char=b - len(text) + len(text.rstrip()))
        _, _, start, end = max(candidates)
        return dict(text=context[a + start:a + end], start_char=a + start, end_char=a + end)

    def predict(self, rows):
        return [self.predict_one(row) for row in rows]


def train_or_restore(splits, tokenizer, max_length, reuse=True):
    seed_everything()
    directory = CHECKPOINT_DIR / str(max_length)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, best_path, resume_path = directory / 'result.json', directory / 'best.pt', directory / 'resume.pt'
    signature = dict(config=CONFIG, max_length=max_length, data=split_signature(splits))
    if reuse and result_path.exists() and best_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['signature'] == signature, 'Checkpoint configuration/data mismatch'
        model = get_model(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f'Restored length {max_length}; original training time {result["training_time_sec"]:.1f}s', flush=True)
        return model, result
    encoded = {name: encode_examples(rows, tokenizer, max_length) for name, rows in splits.items()}
    print('Length', max_length, 'questions', {k: len(v) for k, v in splits.items()},
          'features', {k: len(v[0]) for k, v in encoded.items()}, flush=True)
    model = get_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    steps = math.ceil(len(encoded['train'][0]) / CONFIG['batch_size']) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * CONFIG['warmup_ratio']), steps)
    history, best_f1, best_epoch = [], -1.0, None
    first_epoch, next_batch, elapsed_before, loss_sum, feature_count = 1, 0, 0.0, 0.0, 0
    if reuse and resume_path.exists():
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        assert state['signature'] == signature
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if isinstance(value, torch.Tensor):
                    opt_state[key] = value.to(DEVICE)
        scheduler.load_state_dict(state['scheduler'])
        history, best_f1, best_epoch = state['history'], state['best_f1'], state['best_epoch']
        first_epoch, next_batch, elapsed_before = state['epoch'], state['next_batch'], state['elapsed']
        loss_sum, feature_count = state['loss_sum'], state['feature_count']
        torch.set_rng_state(state['rng'])
        if DEVICE.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print('Resume', max_length, first_epoch, next_batch, flush=True)
    started = perf_counter()

    def save_progress(epoch, next_index):
        atomic_save(dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), history=history, best_f1=best_f1, best_epoch=best_epoch,
            epoch=epoch, next_batch=next_index, loss_sum=loss_sum, feature_count=feature_count,
            elapsed=elapsed_before + perf_counter() - started, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None), resume_path)

    for epoch in range(first_epoch, CONFIG['epochs'] + 1):
        model.train()
        batches = batches_for_epoch(encoded['train'][0], epoch)
        for batch_index, batch in enumerate(make_loader(encoded['train'][0], tokenizer, batches[next_batch:]), start=next_batch):
            optimizer.zero_grad(set_to_none=True)
            output = model(**{k: v.to(DEVICE) for k, v in batch.items()})
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            count = len(batch['input_ids'])
            loss_sum += output.loss.item() * count
            feature_count += count
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(batches):
                print(f'length={max_length} epoch={epoch}/{CONFIG["epochs"]} batch={batch_index+1}/{len(batches)} '
                      f'loss={loss_sum/feature_count:.4f} elapsed={elapsed_before+perf_counter()-started:.0f}s', flush=True)
            if (batch_index + 1) % 50 == 0:
                save_progress(epoch, batch_index + 1)
        prediction, validation_loss = infer_encoded(model, splits['validation'], encoded['validation'], tokenizer)
        scores = evaluate_answers(splits['validation'], prediction)
        history.append(dict(epoch=epoch, training_loss=loss_sum / feature_count,
                            validation_loss=validation_loss, **scores))
        print('VALIDATION', max_length, history[-1], flush=True)
        if scores['f1'] > best_f1:
            best_f1, best_epoch = scores['f1'], epoch
            atomic_save(model.state_dict(), best_path)
        next_batch, loss_sum, feature_count = 0, 0.0, 0
        save_progress(epoch + 1, 0)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = elapsed_before + perf_counter() - started
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    predictions, losses = {}, {}
    for name in ['validation', 'test']:
        predictions[name], losses[name] = infer_encoded(model, splits[name], encoded[name], tokenizer)
    result = dict(signature=signature, max_length=max_length, history=history, best_epoch=best_epoch,
        training_time_sec=elapsed, features={k: len(v[0]) for k, v in encoded.items()},
        predictions=predictions, losses=losses,
        metrics={k: evaluate_answers(splits[k], predictions[k]) for k in predictions},
        hardware=dict(device=str(DEVICE), processor=platform.processor(), torch=torch.__version__, threads=torch.get_num_threads()))
    atomic_json(result, result_path)
    model.eval()
    return model, result
