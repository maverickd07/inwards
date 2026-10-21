"""Signed numeric scoring. Floor is default; truncate is a sensitivity option.
Text/date/entity scoring delegates unchanged to the frozen original scorer.
Numbers require a single numeric expression: no substring or digit-piece matching.
"""
import re,unicodedata,importlib.util
from decimal import Decimal, ROUND_FLOOR, ROUND_DOWN
from pathlib import Path
ORIGINAL=Path(__file__).with_name('exp1_gap_v2_v1.py')
spec=importlib.util.spec_from_file_location('inward_original_scorer',ORIGINAL)
_original=importlib.util.module_from_spec(spec);spec.loader.exec_module(_original)
old_is_correct=_original.is_correct
NUMBER=r'[+−–-]?(?:\d+(?:\.\d+)?|\.\d+)'
FULL=re.compile(r'^\s*('+NUMBER+r')\s*(?:°|degrees?)?\s*([NSEW])?\s*$',re.I)
SCAN=re.compile(r'(?<![\w.])('+NUMBER+r')(?![\w.])')
def norm(s):
    """Normalize typography only; preserve signs, decimal points and symbols."""
    s=unicodedata.normalize('NFKC',str(s)).replace('−','-').replace('–','-')
    s=re.sub(r'<think>.*?</think>',' ',s,flags=re.S)
    s=re.sub(r'(?<!\w)([+-])\s+(?=\d)',r'\1',s)
    return ' '.join(s.casefold().split())
def value(s, allow_prose=False):
    s=norm(s)
    m=FULL.fullmatch(s)
    if m:
        v=Decimal(m[1]);direction=(m[2] or '').upper()
        if direction in ['S','W']:v=-abs(v)
        elif direction in ['N','E']:v=abs(v)
        return v
    if allow_prose:
        ms=list(SCAN.finditer(s))
        # Reject lists, ranges, DMS, ISO numbers in prose, and multiple candidate values.
        if len(ms)==1:
            v=Decimal(ms[0][1])
            direction=re.match(r'\s*(?:°|degrees?)?\s*([NSEW])\b',s[ms[0].end():],re.I)
            if direction:
                v=-abs(v) if direction[1].upper() in ['S','W'] else abs(v)
            return v
    return None

def rounded(v, mode='floor'):
    if mode not in ['floor','truncate']:raise ValueError(mode)
    return v.to_integral_value(rounding=ROUND_FLOOR if mode=='floor' else ROUND_DOWN)

def is_correct(pred,aliases,mode='floor'):
    gold=[value(a) for a in aliases]
    # Numeric mode only when every provided alias is a plain signed number or coordinate.
    if gold and all(v is not None for v in gold):
        p=value(pred,allow_prose=True)
        return p is not None and any(rounded(p,mode)==rounded(g,mode) for g in gold)
    return old_is_correct(pred,aliases)
