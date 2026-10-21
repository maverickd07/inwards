"""Conservative, typed whole-answer scoring; unknown forms remain reviewable.

No substring matching, model/person/seed-specific rules, reference relabeling,
or matching a currency code to a requested currency symbol. The score measures
unambiguously accepted answers under the unchanged benchmark references.
"""
import json
import re
import unicodedata
from pathlib import Path
import scoring_v2_v1 as old

HERE = Path(__file__).resolve().parent
ALIASES = json.loads((HERE / 'aliases.json').read_text())
CATALOG = json.loads((HERE / 'reference_catalog.json').read_text())['values']
SUPPORTED = {'country', 'birthplace_currency', 'birthplace_currency_short',
             'birthplace_currency_symbol', 'birthplace_tld', 'birthplace_ccn3',
             'birthplace_rounded_lat', 'birthplace_rounded_lng', 'birthplace_callingcode'}


def clean(raw):
    s = unicodedata.normalize('NFKC', str(raw)).strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.S).strip()
    if '<think>' in s or '</think>' in s:
        return None
    for _ in range(3):
        for start, end in [('```', '```'), ('**', '**'), ('`', '`'), ('"', '"'), ('“', '”'), ("'", "'")]:
            if len(s) > len(start) + len(end) and s.startswith(start) and s.endswith(end):
                s = s[len(start):-len(end)].strip()
    return ' '.join(s.split())


def unwrap(s, kind):
    # Only fixed answer-introducing phrases; never extract an answer from prose.
    labels = {'country': r'(?:country|birth country)',
              'birthplace_currency': r'currency',
              'birthplace_currency_short': r'(?:currency code|currency abbreviation|ISO 4217 code)',
              'birthplace_currency_symbol': r'(?:currency symbol|symbol)',
              'birthplace_tld': r'(?:top-level domain|TLD)',
              'birthplace_ccn3': r'(?:ISO 3166-1 numeric code|numeric code)',
              'birthplace_callingcode': r'(?:calling code|country calling code)'}
    s = re.sub(r'^answer\s*:\s*', '', s, flags=re.I)
    if kind in labels:
        s = re.sub(r'^(?:the\s+)?' + labels[kind] + r'\s*(?::|is\b)\s*', '', s, flags=re.I)
    return s.strip()


def canonical(s, kind):
    if kind in ('country', 'birthplace_currency'):
        s = unicodedata.normalize('NFKD', s.casefold())
        s = ''.join(c for c in s if not unicodedata.combining(c))
        s = s.replace('&', ' and ').replace('.', '')
        s = re.sub(r"[’'\-]", ' ', s)
        s = ' '.join(s.split()).rstrip('.,;:!')
        return re.sub(r'^the\s+', '', s)
    if kind == 'birthplace_currency_short':
        s = s.rstrip('.').upper()
        return s if re.fullmatch(r'[A-Z]{3}', s) else None
    if kind == 'birthplace_currency_symbol':
        # Keep punctuation and the entire symbol; B/. is not B$ and Br is not Braille.
        return s.casefold()
    if kind == 'birthplace_tld':
        s = s.casefold()
        if s.startswith('.'):
            s = s[1:]
        if s.endswith('.'):
            s = s[:-1]
        if not re.fullmatch(r'[^\W_]+(?:-[^\W_]+)*', s, flags=re.U):
            return None
        try:
            return s.encode('idna').decode('ascii')
        except UnicodeError:
            return None
    if kind == 'birthplace_ccn3':
        # Codes are integers, not coordinates; do not floor a fractional code.
        return str(int(s)) if re.fullmatch(r'[0-9]{1,3}', s) else None
    if kind == 'birthplace_callingcode':
        return str(int(s.lstrip('+'))) if re.fullmatch(r'\+?[0-9]{1,4}', s) else None
    raise ValueError('Unsupported canonicalization: ' + kind)


def accepted(aliases, kind):
    result = {canonical(clean(a), kind) for a in aliases if clean(a)} - {None, ''}
    for group in ALIASES.get(kind, []):
        norms = {canonical(clean(a), kind) for a in group}
        if result & norms:
            result |= norms
    return result


def score(pred, aliases, category, metric='composed'):
    if metric not in ('composed', 'q1', 'q2'):
        raise ValueError(metric)
    kind = 'country' if metric == 'q1' else category
    if kind not in SUPPORTED:
        raise ValueError('Unsupported answer type: ' + kind)
    def result(correct, reason, review=False):
        return {'correct': bool(correct), 'reason': reason, 'needs_review': review,
                'answer_type': kind, 'reference_policy': 'unchanged_dataset_snapshot'}
    if not aliases:
        raise ValueError('Empty reference aliases')
    s = clean(pred)
    if s is None:
        return result(False, 'unclosed_reasoning_markup', True)
    if not s:
        return result(False, 'empty_answer')
    if kind in ('birthplace_rounded_lat', 'birthplace_rounded_lng'):
        return result(old.is_correct(pred, aliases), 'frozen_signed_floor_numeric_rule')
    s = unwrap(s, kind)
    want = accepted(aliases, kind)
    if not want:
        raise ValueError('Unparseable references: ' + str(aliases))
    value = canonical(s, kind)
    if value is not None and value in want:
        return result(True, 'whole_answer_or_declared_alias')
    if kind == 'birthplace_tld':
        tokens = re.findall(r'(?<!\w)\.[^\W_]+', s.casefold())
        if len(set(tokens)) > 1:
            return result(False, 'multiple_domain_answers', True)
    if value is None:
        return result(False, 'unsupported_or_multiple_answer', True)
    known = accepted(CATALOG.get(kind, []), kind)
    if value in known:
        return result(False, 'different_reference_answer')
    if kind in ('birthplace_ccn3', 'birthplace_callingcode', 'birthplace_currency_short', 'birthplace_tld'):
        return result(False, 'different_typed_answer')
    # Unknown name/symbol/prose may be a wrong answer or an omitted legitimate
    # alias. Preserve it for independent review rather than claiming factual error.
    return result(False, 'unmatched_name_symbol_or_prose', True)
