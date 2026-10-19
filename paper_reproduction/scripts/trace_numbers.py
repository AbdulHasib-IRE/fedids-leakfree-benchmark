# Trace every number in main.tex to a source value (export CSV/JSON, derived facts, TOST, notebook outputs).
import re, json, glob, sys
tex = open(sys.argv[1]).read()
body = tex[tex.index('\\begin{abstract}'):tex.index('\\bibliography')]
body = re.sub(r'\\(label|ref|eqref|subref|cite[pt]?|includegraphics(\[[^\]]*\])?|begin|end)\{[^}]*\}', ' ', body)
body = re.sub(r'\\(?:sisetup|usepackage)[^\n]*', ' ', body)
nums = set(re.findall(r'(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w])', body))
src_txt = ''
for f in sum((glob.glob(a + "/*") for a in [sys.argv[2], sys.argv[3]] + sys.argv[5:]), []):
    if f.endswith(('.csv', '.json', '.txt')): src_txt += open(f).read() + '\n'
nb = json.load(open(sys.argv[4])) if sys.argv[4] != '-' else {'cells': []}
for c in nb['cells'][172:]:
    for o in c.get('outputs', []): src_txt += ''.join(o.get('text', '')) + '\n'
for c in nb['cells'][6:8]:
    for o in c.get('outputs', []): src_txt += ''.join(o.get('text', '')) + '\n'
vals = set()
for s in re.findall(r'-?\d+\.\d+(?:e-?\d+)?|-?\d+', src_txt) + re.findall(r'\d{1,3}(?:,\d{3})+(?![\d.])', src_txt):
    try: v = float(s.replace(',', ''))
    except: continue
    v = abs(v)
    for d in range(0, 5):
        vals.add(f'{round(v, d):.{d}f}'); vals.add(f'{round(v * 100, d):.{d}f}')
unmatched = []
for n in sorted(nums):
    k = n.replace(',', '')
    if k in vals: continue
    if '.' not in k and int(k) <= 20: continue          # small integers: counts, section text
    unmatched.append(n)
print(len(nums), 'numbers;', len(unmatched), 'unmatched:', unmatched)
