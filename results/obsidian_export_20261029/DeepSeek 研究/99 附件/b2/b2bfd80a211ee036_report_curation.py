from pathlib import Path
import json,collections,datetime
from collect_public import api,qids,save
P=Path(__file__).resolve().parent
s=json.loads((P/'curation_final'/ 'CURATION_SUMMARY.json').read_text());a=json.loads((P/'curation_final'/ 'CURATION_AUDIT.json').read_text());chosen=json.loads((P/'curation_final'/ 'SELECTED_NEW400.json').read_text());ens=json.loads((P/'PERSON_ENTITIES_EXPANDED.json').read_text());assert s['selection_complete']
occ=collections.Counter(x for r in chosen for x in qids(ens[r['qid']],'P106'));labels={};ids=sorted(occ)
for i in range(0,len(ids),40):
 d=api('www.wikidata.org',{'action':'wbgetentities','ids':'|'.join(ids[i:i+40]),'props':'labels','languages':'en|zh'})
 for q,e in d['entities'].items():labels[q]=e.get('labels',{}).get('zh',e.get('labels',{}).get('en',{})).get('value',q)
save('SELECTED_OCCUPATIONS.json',{'overlapping_counts':dict(occ),'labels':labels})
translations={'not_confirmed_human':'非已确认人物','identity_or_english_biography_missing':'身份或英文传记缺失','ambiguous_shared_display_name':'展示名重名歧义','existing_person_or_alias':'既有人物实体或别名重叠','evaluation_entity_or_alias_leakage':'评估人物实体或别名重叠','birth_date_missing_or_conflicting':'出生年缺失或冲突','birth_year_outside_candidate_window':'实际出生年不在1985–1999','old_person_before_conservative_1955_window':'旧人物出生早于保守1955时间窗','uncertain_birthplace_or_citizenship_statement':'出生地或国籍存在未知声明','multiple_citizenship_recorded':'明确记录多个公民身份','citizenship_missing':'公民身份记录缺失','citizenship_outside_conservative_scope':'公民国在保守范围外','birthplace_missing_or_multiple':'出生地缺失或多值','uncertain_birth_country_statement':'出生国存在未知声明','historical_or_multiple_birth_country':'出生地点记录历史或多个国家','birth_country_missing_or_not_loaded_after_prior_exclusion':'出生国缺失或此前已排除而未继续取地点资料','birth_country_outside_conservative_scope':'出生国在保守范围外','germany_before_1991':'德国出生时间早于1991','citizenship_differs_from_birth_country':'公民国与出生国不同（非双国籍证明）','territorial_or_historical_location_risk':'历史/争议/海外地区地点风险','old_reference_birth_country_not_confirmed':'旧参考出生国未获核验','not_even_md5_training_partition':'不属预定MD5偶数训练分区','original_identity_unresolved':'旧人物实体未解析','biography_unavailable':'完整传记未取得','existing_person_redirect_alias':'既有人物重定向别名交集','biography_multiple_or_naturalised_citizenship_signal':'传记多国籍/入籍或复合身份信号','eligible_not_selected_by_fixed_cap':'资料合格但固定国家轮转/上限未选中'}
lines=['exp17 新增400人物筛查报告（仅准备阶段）',f"新候选 {s['new']['audited']}，资料及分区筛查合格 {s['new']['eligible']}，固定选择400。",f"旧216人物资料复核：合格 {s['old']['selected']}，剔除 {216-s['old']['selected']}；历史数据保持不动。",'选择未读取新增模型预测。400人×4类别=1600个候选题，实际严格配对数尚未计算。','先前350–480题的粗估未包含本轮更严格人口筛查，不能作为本轮保证。','', '不重叠的主排除计数（每人只按首个原因计一次）：']
for mode in ['new','old']:
 lines.append('新增候选' if mode=='new' else '原训练人物')
 for reason,n in s[mode]['exclusive_exclusions'].items():lines.append(f'  {translations.get(reason,reason)}：{n}')
 lines.append('可重叠风险标记：'+json.dumps(s[mode]['overlapping_risk_flags'],ensure_ascii=False))
lines+=['','最终400人出生国：']+[f'{c}：{n}' for c,n in sorted(s['selected_countries'].items())]
lines+=['','主要职业声明（可重叠，并非互斥职业分类）：']+[f'{labels[q]} ({q})：{n}' for q,n in occ.most_common(15)]
lines+=['','评估隔离：原A/C/D合计252名评估人物已全部解析实体；排除全部原数据人物实体、名称、已取英文别名及完整重定向分页。','公开资料采用自动结构化与传记信号筛查；不是逐人线下身份认证。', '第一窗口不足400：追加同一1985–1999年份分类随后200条/年；筛选规则与最终400人上限不变。', '公开资料来源：https://www.wikidata.org/ 与 https://en.wikipedia.org/；逐人实体链接、来源版本和声明ID见inputs/new_people.jsonl及curation_audit.json。','初始候选窗为1985–1999出生分类每年最近加入500条，非随机总体样本；保守国家范围与年代限制见封存细则。','明确多国籍记录、传记风险信号、出生国冲突、范围外和时间窗外分别统计。不能把全部排除者称为双重国籍或出生于争议地区。','公开数据库和传记可能遗漏第二国籍；本审核表示通过已列明的资料交叉检查，不是法律上的无第二国籍证明。','传记关键词可能命中亲属或背景，这些也保守排除，不能计作已确认本人双国籍。','旧人物也额外接受保守年代与身份审查，因此本池不等同于历史864题全部。','每组严格配对不足300题，程序完成损失统计并停止询问用户；达到300也不训练。','本阶段没有训练效应结果。']
with (P/'AUDIT_ZH.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
print('Curation report complete')
