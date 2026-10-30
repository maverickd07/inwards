---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp17_inference_pool_20261016_upload_verification_AUDIT_ZH.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/01/01f251a089820f02_AUDIT_ZH.txt|AUDIT_ZH.txt]]

exp17 新增400人物筛查报告（仅准备阶段）
新候选 10353，资料及分区筛查合格 762，固定选择400。
旧216人物资料复核：合格 15，剔除 201；历史数据保持不动。
选择未读取新增模型预测。400人×4类别=1600个候选题，实际严格配对数尚未计算。
先前350–480题的粗估未包含本轮更严格人口筛查，不能作为本轮保证。

不重叠的主排除计数（每人只按首个原因计一次）：
新增候选
  出生地缺失或多值：831
  出生年缺失或冲突：1384
  不属预定MD5偶数训练分区：795
  公民国在保守范围外：2472
  公民身份记录缺失：2379
  出生地点记录历史或多个国家：447
  资料合格但固定国家轮转/上限未选中：362
  传记多国籍/入籍或复合身份信号：51
  身份或英文传记缺失：262
  明确记录多个公民身份：314
  历史/争议/海外地区地点风险：331
  德国出生时间早于1991：79
  展示名重名歧义：18
  出生国在保守范围外：29
  实际出生年不在1985–1999：101
  出生地或国籍存在未知声明：3
  公民国与出生国不同（非双国籍证明）：27
  既有人物实体或别名重叠：17
  完整传记未取得：7
  非已确认人物：32
  出生国缺失或此前已排除而未继续取地点资料：8
  died_before_adulthood：4
可重叠风险标记：{"historical_or_multiple_birth_country": 652, "territorial_or_historical_location_risk": 956, "biography_citizenship_signal": 51, "multiple_citizenship_recorded": 345, "changed_country_time_window": 86, "cross_country_identity_risk": 80}
原训练人物
  旧人物出生早于保守1955时间窗：74
  出生国在保守范围外：3
  明确记录多个公民身份：52
  公民身份记录缺失：4
  出生地点记录历史或多个国家：1
  公民国在保守范围外：43
  出生年缺失或冲突：14
  出生地缺失或多值：2
  传记多国籍/入籍或复合身份信号：2
  公民国与出生国不同（非双国籍证明）：2
  非已确认人物：2
  历史/争议/海外地区地点风险：1
  评估人物实体或别名重叠：1
可重叠风险标记：{"multiple_citizenship_recorded": 88, "cross_country_identity_risk": 7, "territorial_or_historical_location_risk": 7, "historical_or_multiple_birth_country": 8, "biography_citizenship_signal": 2}

最终400人出生国：
Argentina：27
Australia：24
Austria：3
Brazil：27
Canada：27
Chile：9
Colombia：11
France：27
Germany：10
Greece：3
Iceland：3
Italy：26
Japan：27
Mexico：20
New Zealand：3
Norway：15
Peru：15
Portugal：4
South Korea：25
Spain：26
Sweden：26
Switzerland：5
United Kingdom：2
United States：26
Uruguay：9

主要职业声明（可重叠，并非互斥职业分类）：
足球运动员 (Q937857)：90
政治人物 (Q82955)：69
演員 (Q33999)：67
歌手 (Q177220)：48
電影演員 (Q10800557)：32
电视演员 (Q10798782)：28
模特儿 (Q4610556)：23
律師 (Q40348)：12
田径运动员 (Q11513337)：12
詞曲作家 (Q753110)：11
足球裁判 (Q859528)：10
饶舌歌手 (Q2252262)：10
音樂家 (Q639669)：9
聲優 (Q622807)：9
創作歌手 (Q488205)：7

评估隔离：原A/C/D合计252名评估人物已全部解析实体；排除全部原数据人物实体、名称、已取英文别名及完整重定向分页。
公开资料采用自动结构化与传记信号筛查；不是逐人线下身份认证。
第一窗口不足400：追加同一1985–1999年份分类随后200条/年；筛选规则与最终400人上限不变。
公开资料来源：https://www.wikidata.org/ 与 https://en.wikipedia.org/；逐人实体链接、来源版本和声明ID见inputs/new_people.jsonl及curation_audit.json。
初始候选窗为1985–1999出生分类每年最近加入500条，非随机总体样本；保守国家范围与年代限制见封存细则。
明确多国籍记录、传记风险信号、出生国冲突、范围外和时间窗外分别统计。不能把全部排除者称为双重国籍或出生于争议地区。
公开数据库和传记可能遗漏第二国籍；本审核表示通过已列明的资料交叉检查，不是法律上的无第二国籍证明。
传记关键词可能命中亲属或背景，这些也保守排除，不能计作已确认本人双国籍。
旧人物也额外接受保守年代与身份审查，因此本池不等同于历史864题全部。
每组严格配对不足300题，程序完成损失统计并停止询问用户；达到300也不训练。
本阶段没有训练效应结果。


返回 [[00 研究首页]]
