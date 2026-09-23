"""Fix the quote-matching regex line in scoring.py and notebook cell 10 via line rewriting."""
import json
from pathlib import Path

# The correct regex as actual characters: r"[\u201c"]([^\u201d"]{10,})[\u201d"]"
# Using explicit unicode chars for curly quotes and escaped straight quote.
GOOD_LINE = '    quotes = re.findall(r"[\u201c"]([^\u201d"]{10,})[\u201d"]", ev)'

sp = Path('data/blind_study/scoring.py')
lines = sp.read_text(encoding='utf-8').splitlines(keepends=True)
fixed = 0
for i, l in enumerate(lines):
    if 're.findall' in l and '201c' in l:
        lines[i] = GOOD_LINE + '\n'
        fixed += 1
sp.write_text(''.join(lines), encoding='utf-8')
print('scoring.py lines fixed:', fixed)

nb = json.loads(Path('notebooks/06_blind_explanation_quality.ipynb').read_text(encoding='utf-8'))
cell = nb['cells'][10]
src = ''.join(cell['source'])
fixed_nb = 0
if 're.findall' in src and '201c' in src:
    import re
    m = re.search(r'.*re\.findall.*201c.*\n', src)
    if m:
        old_line = m.group(0)
        # in the notebook, this line is inside the triple-quoted scorer_src string;
        # keep its leading indent as-is
        indent = old_line[:len(old_line) - len(old_line.lstrip())]
        new_line = (indent + 'quotes = re.findall(r"[\\u201c"]([^\\u201d"]{10,})[\\u201d"]", ev)\n')
        # NOTE: inside the notebook string, backslashes must be doubled for the regex
        new_line = indent + 'quotes = re.findall(r"[\\\\u201c\\"]([^\\\\u201d\\"]{10,})[\\\\u201d\\"]", ev)\n'
        src = src.replace(old_line, new_line)
        cell['source'] = src.splitlines(keepends=True)
        fixed_nb = 1
Path('notebooks/06_blind_explanation_quality.ipynb').write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('notebook lines fixed:', fixed_nb)

# validate both parse
import ast
ast.parse(sp.read_text(encoding='utf-8'))
nb2 = json.loads(Path('notebooks/06_blind_explanation_quality.ipynb').read_text(encoding='utf-8'))
for c in nb2['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
print('both parse OK')
