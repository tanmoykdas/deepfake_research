import json
from pathlib import Path
p=Path(r'f:/Sixth Semester/DEEPfake Papers/FakeBDTeen/Notebooks/kaggle_videoswin_wavlm_multimodal_training.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
cells=nb.get('cells',[])
code=[c for c in cells if c.get('cell_type')=='code']
md=[c for c in cells if c.get('cell_type')=='markdown']
print(f'cells_total={len(cells)} code={len(code)} markdown={len(md)}')
miss=[i for i,c in enumerate(cells,1) if 'language' not in c.get('metadata',{})]
print(f'cells_missing_metadata_language={len(miss)}')
import re
pat=re.compile(r'^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b')
names={}
for c in code:
    for ln in c.get('source',[]):
        m=pat.match(ln.strip())
        if m:
            names[m.group(2)]=names.get(m.group(2),0)+1
dups=sorted([k for k,v in names.items() if v>1])
print(f'duplicate_top_level_names={len(dups)}')
print('duplicate_names_sample=',dups[:30] if dups else [])
text='\n'.join('\n'.join(c.get('source',[])) for c in code)
req=['reserve_held_out_speakers','get_llrd_param_groups','ModelEMA','stage_0_contrastive','evaluate_modality_ablations','compute_calibration','log_attention_weights','run_pilot_training_pipeline']
missing=[x for x in req if x not in text]
print(f'required_markers_missing={len(missing)}')
print('missing_markers=',missing)
