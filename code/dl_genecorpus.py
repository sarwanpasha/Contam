import os, sys
from huggingface_hub import hf_hub_download
REPO='ctheodoris/Genecorpus-30M'
DEST=os.environ['WORK']+'/data/reference'
os.makedirs(DEST, exist_ok=True)
small=['token_dictionary.pkl','genecorpus_30M_2048_lengths.pkl','genecorpus_30M_2048_sorted_lengths.pkl','example_input_files/gene_info_table.csv']
big=['genecorpus_30M_2048.dataset/dataset.arrow','genecorpus_30M_2048.dataset/dataset_info.json','genecorpus_30M_2048.dataset/state.json']
targets = small if sys.argv[1]=='small' else big
for f in targets:
    try:
        p=hf_hub_download(repo_id=REPO, filename=f, repo_type='dataset', local_dir=DEST)
        print('OK', f, '->', p, flush=True)
    except Exception as e:
        print('FAIL', f, type(e).__name__, e, flush=True)
print('=== DONE ===')
