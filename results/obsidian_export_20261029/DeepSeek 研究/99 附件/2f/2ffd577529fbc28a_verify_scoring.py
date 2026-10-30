from scoring_v2 import is_correct,old_is_correct
checks=[('-2',[2],False),('2.5',[2],True),('2.5',[5],False),('-2.5',[-3],True),('-2.5',[2],False),('−2',[-2],True),('- 2',[2],False),('2° S',[-2],True),('latitude is 2° S',[-2],True),('2° N',[2],True),('2 or 3',[2],False),('22',[2],False),('004',['4'],True),('+249',['249'],True),('Paris',['Paris'],True),('December 28, 1934',['December 28, 1934'],True),('The code is 458',['458'],True)]
for pred,gold,expected in checks:
 assert is_correct(pred,gold)==expected,(pred,gold,is_correct(pred,gold),expected)
assert is_correct('-2.5',[-2],mode='truncate')
assert not is_correct('-2.5',[-2],mode='floor')
assert old_is_correct('-2',[2]) and old_is_correct('2.5',[5])
print('PASS: signed decimals, floor/truncate, Unicode minus, compass direction, multiple numbers, whole number matching, codes and text/date regression')
