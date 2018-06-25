
# .... meant to be stuck on the end of cifar10.py ...
# but this doesn't give great results w/ linear-linear

from glob import glob
from tqdm import tqdm

# Load weights
weights = {}
for f in tqdm(glob('weights/*')):
    weights[int(f.split('-')[-1])] = torch.load(f)

res = []
for n in range(4, len(weights) + 1):
    try:
        new_weights = {}
        for k in weights[0].keys():
            new_weights[k] = torch.stack([weights[i][k] for i in range(n - 3, n)]).mean(dim=0)
            
        model.load_state_dict(new_weights)
        
        res.append({
            "n"   : n,
            "acc" : float(model.eval_epoch(dataloaders, mode='test')['acc'])
        })
        print(res[-1])
    except:
        pass


ks = list(weights.keys())[-10:]

res = []
for _ in range(10):
    sks = tuple(sorted(np.random.choice(ks, 5)))
    
    new_weights = {}
    for k in weights[0].keys():
        new_weights[k] = torch.stack([weights[i][k] for i in sks]).mean(dim=0)
    
    model.load_state_dict(new_weights)
    
    res.append({
        "sks" : sks,
        "acc" : float(model.eval_epoch(dataloaders, mode='test')['acc'])
    })
    print(res[-1])
