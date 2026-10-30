"""Frozen prompts: control selection uses token lengths only, before generation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'inputs'))
from exp1_gap_v2_v1 import SYSTEM,clear_question,ctx_question
CONDITIONS=['C0','C1','C1-ctrl','C2','C3']
POINTER='Before answering, recall which country this person was born in.'
# Length and fixed grammar only, never outputs; first exact match wins.
CONTROLS=[
 'Before answering, recall that the question is about a country.',
 'Before answering, recall that this question is about a country.',
 'Before answering, recall that the question asks about a country.',
 'Before answering, recall that this question asks about a country.',
 'Before answering, recall that this question is about one country.',
 'Before answering, recall that the current question concerns a country.',
 'Before answering, recall that the following question concerns a country.',
 'Before answering, recall that the question below concerns a country.',
 'Before answering, recall that the question is about a particular country.',
 'Before answering, recall that this question is about a particular country.',
]
C2_SYSTEM='Answer the question in exactly two lines: Country: <the country where the person was born>\nAnswer: <the final answer>. No explanation.'
def messages(e,c,ctrl):
 q=clear_question(e)
 if c=='C1':q=POINTER+'\n'+q
 elif c=='C1-ctrl':q=ctrl+'\n'+q
 elif c=='C3':q=ctx_question(e)
 return [{'role':'system','content':C2_SYSTEM if c=='C2' else SYSTEM},{'role':'user','content':q}]
def render(tok,e,c,ctrl):
 return tok.apply_chat_template(messages(e,c,ctrl),tokenize=False,add_generation_prompt=True,enable_thinking=False)
def choose_control(tok,rows):
 for ctrl in CONTROLS:
  if all(len(tok.encode(render(tok,r['e'],'C1',ctrl),add_special_tokens=False))==len(tok.encode(render(tok,r['e'],'C1-ctrl',ctrl),add_special_tokens=False)) for r in rows):return ctrl
 raise ValueError('No exact full-template length-matched control. STOP before generation.')
