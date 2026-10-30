"""Synthetic cases only; no tuning against aggregate model scores."""
import json
import unittest
from pathlib import Path
from scorer_v3 import score, ALIASES, canonical


class ScorerTests(unittest.TestCase):
    def test_boundary_and_symbol_regressions(self):
        for pred, gold in [('Braille pattern dots-123456','Br'), ('B$','B/.'),
                           ('US$','$'), ('EUR','€'), ('JPY','¥'), ('Brazil','Br')]:
            self.assertFalse(score(pred,[gold],'birthplace_currency_symbol')['correct'])
        for symbol in ['Br','B/.','€','$','Ft','د.إ','₮']:
            self.assertTrue(score(symbol,[symbol],'birthplace_currency_symbol')['correct'])

    def test_no_negation_or_contradiction_fragment(self):
        for pred in ['.com (Uruguay’s TLD is .uy)', '.uy or .com', 'not .uy',
                     '.com, .uy', 'The TLD is .com, not .uy', 'The TLD is not .uy']:
            result = score(pred,['.uy'],'birthplace_tld')
            self.assertFalse(result['correct'])
            self.assertTrue(result['needs_review'])
        for pred in ['not Euro','Euro or US dollar','Euro (USD)', 'not the United Kingdom']:
            self.assertFalse(score(pred,['Euro'],'birthplace_currency')['correct'])
        self.assertFalse(score('Democratic Republic of the Congo',['Republic of the Congo'],'x','q1')['correct'])
        self.assertFalse(score('Great Britain',['United Kingdom'],'x','q1')['correct'])

    def test_country_and_currency_aliases(self):
        cases = [('Pound sterling','British pound','birthplace_currency','composed'),
                 ('UAE Dirham','United Arab Emirates dirham','birthplace_currency','composed'),
                 ('Russian rouble','Russian ruble','birthplace_currency','q2'),
                 ('US dollar','United States dollar','birthplace_currency','composed'),
                 ('USA','United States','birthplace_tld','q1'),
                 ('Türkiye','Turkey','birthplace_tld','q1')]
        for pred, gold, category, metric in cases:
            self.assertTrue(score(pred,[gold],category,metric)['correct'])
        self.assertFalse(score('dollar',['Canadian dollar'],'birthplace_currency')['correct'])
        self.assertFalse(score('pound',['British pound'],'birthplace_currency')['correct'])

    def test_wrappers_and_domain_labels(self):
        for pred in ['.uy','uy','.UY','`.uy`','**.uy**','TLD: .uy','The top-level domain is .uy.']:
            self.assertTrue(score(pred,['.uy'],'birthplace_tld')['correct'],pred)
        self.assertTrue(score('xn--p1ai',['.рф'],'birthplace_tld')['correct'])
        self.assertTrue(score('الاردن.',['الاردن.'],'birthplace_tld')['correct'])
        for pred in ['The currency is Euro.', 'Answer: Euro', '"Euro"', '<think>guess</think>Euro']:
            self.assertTrue(score(pred,['Euro'],'birthplace_currency')['correct'])
        self.assertFalse(score('<think>Euro',['Euro'],'birthplace_currency')['correct'])

    def test_codes_not_coordinates(self):
        for pred in ['8','008','Numeric code: 008']:
            self.assertTrue(score(pred,['008'],'birthplace_ccn3')['correct'])
        for pred in ['840.9','+840','840 or 124','840/124','ISO 3166-1: 840']:
            self.assertFalse(score(pred,['840'],'birthplace_ccn3')['correct'])
        self.assertFalse(score('+27',['710'],'birthplace_ccn3')['correct'])
        self.assertTrue(score('eur',['EUR'],'birthplace_currency_short')['correct'])
        self.assertFalse(score('EUR/USD',['EUR'],'birthplace_currency_short')['correct'])
        self.assertFalse(score('Euro',['EUR'],'birthplace_currency_short')['correct'])
        self.assertTrue(score('-34.2',['-35'],'birthplace_rounded_lat')['correct'])
        self.assertFalse(score('-34.2',['-34'],'birthplace_rounded_lat')['correct'])

    def test_reference_gold_roundtrip(self):
        # Every frozen D alias should match itself, for all three metrics.
        local = json.loads((Path(__file__).parent / 'local_inputs.json').read_text())
        rows = [json.loads(s) for s in Path(local['plan']).read_text().splitlines()]
        for row in rows:
            for metric, field in [('composed','Answer'),('q1','A1'),('q2','A2')]:
                for alias in row['e'][field]:
                    self.assertTrue(score(alias,row['e'][field],row['category'],metric)['correct'],(metric,alias))

    def test_alias_groups_disjoint(self):
        for kind, groups in ALIASES.items():
            seen = set()
            for group in groups:
                norms = {canonical(a,kind) for a in group}
                self.assertFalse(seen & norms)
                seen |= norms

    def test_typed_empty_and_unknown(self):
        self.assertFalse(score('',['Euro'],'birthplace_currency')['correct'])
        self.assertTrue(score('novel currency phrase',['Euro'],'birthplace_currency')['needs_review'])
        with self.assertRaises(ValueError):
            score('x',['x'],'unsupported_category')


if __name__ == '__main__':
    unittest.main()
