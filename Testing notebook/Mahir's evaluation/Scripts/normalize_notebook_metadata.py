import json
from pathlib import Path
p=Path(r'f:/Sixth Semester/DEEPfake Papers/FakeBDTeen/Notebooks/kaggle_videoswin_wavlm_multimodal_training.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
changed=0
for c in nb.get('cells',[]):
    meta=c.setdefault('metadata',{})
    if 'language' not in meta:
        meta['language']=('python' if c.get('cell_type')=='code' else 'markdown')
        changed+=1
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1),encoding='utf-8')
print('cells_updated=',changed)
