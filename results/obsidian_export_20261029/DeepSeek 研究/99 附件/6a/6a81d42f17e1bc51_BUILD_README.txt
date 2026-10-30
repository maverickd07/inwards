向内破：exp1–exp11 最终阶段报告
版本：2026-10-11

入口：inward_final_zh.pdf（14 页 A4；4 张矢量图、16 张表，含来源映射表）
作者：陈威宇（Daniel Chen），Bocconi × HEC Paris
源文件：inward_final_zh.tex

构建说明
1. build_evidence.py：从研究根目录的历史文件及 exp11_685985_review.tgz 读取证据，核验、解包到新的 raw_exp11/ 并生成 evidence.json 和四张矢量图。
   依赖：Python、numpy、matplotlib；需要原研究工作区，压缩交付包不包括全部历史实验原件。
   本次通过：旧来源 40 个哈希不变；exp11 21 次拟合；94,500 个训练后评分标签；C 配对检验重算；exp9/10 C 组逐题汇总校验。
2. build_report.py：仅使用 evidence.json 生成 LaTeX；可在交付包中独立运行。
3. 编译：在本目录使用 xelatex -interaction=nonstopmode -halt-on-error inward_final_zh.tex 两次；或 tectonic --keep-logs inward_final_zh.tex。
   本次使用已安装的 Tectonic（XeTeX 引擎），没有安装新 TeX 发行版；不声称执行过 xelatex 命令。
   中文字体：FandolSong Regular/Bold（宋体衬线字，标准 TeX 字体资源），最终 PDF 已嵌入全部字体；西文字体：TeX Gyre Pagella。
   压缩包不另分发字体文件；编译需要发行版中对应字体。
4. verify_pdf.py：依赖 pypdfium2、pypdf。核验 14 页、字体嵌入与中文提取，并生成 qa/page-*.png。

排版验证
- 原生 LaTeX 编辑器编译成功。编辑器不支持本地多文件图表资源时显示说明占位；交付 PDF 是包含四张完整矢量图的外部编译版本。
- 最终 PDF 逐页渲染检查；所有图与表顺序连续，未将正文句子交给跨页浮动图表分割。
- 18 个 PDF 字体资源均嵌入；PDFium 中文提取通过；无 overfull hbox。
- 仅历史待核实表一处 underfull hbox（宽松字距），不截断内容。
- Poppler 在本机缺少默认 CJK 映射路径时不能正确渲染；最终视觉检查使用 PDFium，中文显示和提取正常。

文件说明
evidence.json：可复用的核验数据快照。
source_sha256.json：原始输入和关键派生来源的文件哈希。
SOURCE_MAP.txt：每张图表到来源文件/字段的映射。
DECISIONS.txt：对用户所附修改意见的采纳与修正。
raw_exp11/：下载归档的新副本；没有覆盖旧实验。
qa/：最终渲染与 verification.json。

研究边界
本文没有重跑 GPU 实验，没有把 exp3–exp6 缺失原始结果的数字当成已核验结论，没有启动 exp12。来源索引及待核实项目见报告附录。不能把该报告当成独立实验室复现或全新封存测试集上的一次性确认。
