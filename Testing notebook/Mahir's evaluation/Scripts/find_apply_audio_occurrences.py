import json
from pathlib import Path
p=Path(r'f:/Sixth Semester/DEEPfake Papers/FakeBDTeen/Notebooks/kaggle_videoswin_wavlm_multimodal_training.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
for i,c in enumerate(nb.get('cells',[]),1):
    src=''.join(c.get('source',[]))
    if 'apply_audio_augmentations' in src:
        print('---')
        print('cell_index=',i)
        print('cell_type=',c.get('cell_type'))
        print('cell_id=',c.get('metadata',{}).get('id','<no-id>'))
        for ln in src.splitlines():
            if 'apply_audio_augmentations' in ln:
                print('    ',ln)
        print('\n')
