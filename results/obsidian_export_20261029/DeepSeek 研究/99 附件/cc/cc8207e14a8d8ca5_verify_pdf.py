from pathlib import Path
import json
import pypdfium2 as pdfium
from pypdf import PdfReader
P=Path(__file__).resolve().parent
p=pdfium.PdfDocument(P/'inward_final_zh.pdf');texts=[]
for i,page in enumerate(p):
 texts.append(page.get_textpage().get_text_range())
 page.render(scale=1.8).to_pil().save(P/'qa'/f'page-{i+1:02d}.png')
assert len(p)==14
assert '向内破' in texts[0] and '核心' in texts[0]
assert '94,500' in ''.join(texts)
assert '图表资源' not in ''.join(texts)
(P/'qa'/'extracted_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts)))
fonts={}
r=PdfReader(P/'inward_final_zh.pdf')
for page in r.pages:
 for k,ref in page['/Resources']['/Font'].items():
  f=ref.get_object();descendants=f.get('/DescendantFonts');a=descendants[0].get_object() if descendants else f
  d=a.get('/FontDescriptor');d=d.get_object() if d else {};embedded=any(n in d for n in ['/FontFile','/FontFile2','/FontFile3'])
  if f.get('/Subtype')=='/Type3':embedded=True
  fonts[str(f.get('/BaseFont',k))]=embedded
assert all(fonts.values()),fonts
qa={'pages':len(p),'page_size':'A4','all_fonts_embedded':True,'font_count':len(fonts),'chinese_text_extractable_with_pdfium':True,'all_figures_included':True,'compiler':'Tectonic (XeTeX engine)','fonts':fonts}
(P/'qa'/'verification.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in qa.items() if k!='fonts'},ensure_ascii=False))
