"""Small execution tests for transfer isolation, frozen states and selection."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile

from search_replacement_grid import read,write,sha,identity
from run_transfer_robustness import freeze_selection


def main():
    import numpy as np
    import torch
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'PINN4SOH'))
    from Model.Model import Solution_u,MLP
    from run_transfer import state_hash
    with tempfile.TemporaryDirectory(prefix='transfer_robustness_fixture_') as temp:
        temp=Path(temp);files=[]
        for index in range(3):
            path=temp/('battery%d.csv'%index);n=35+index*3;t=np.arange(n)
            features=np.stack([np.sin((t+index+1)*(i+1)/45)+t/100 for i in range(16)],axis=1)
            np.savetxt(path,np.column_stack([features,1.1*(1-t*.001)]),delimiter=',',comments='',
                header=','.join(['feature%d'%i for i in range(16)]+['capacity']))
            files.append(str(path))
        torch.manual_seed(11);u=Solution_u();g=MLP(input_dim=35,output_dim=1,layers_num=3,hidden_dim=60)
        checkpoint=temp/'source.pth';torch.save(dict(solution_u=u.state_dict(),dynamical_F=g.state_dict()),checkpoint)
        source=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),historical_training_seed=11,
            source_repeat=1,source_record=dict(fixture_only=True))
        base=dict(format='transfer-robustness-worker-v1',lane='fixture',parameters=dict(lr=.001,alpha=.1,beta=.01,
            freeze='all_u',mono='mean_eval',weight_decay=0,initialization='source'),candidate_id='fixture',mode='train',
            seed=23,source_repeat=1,stack='modern',device='cpu',repo=str(root/'PINN4SOH'),protocol_sha256='synthetic',
            fixed=dict(epochs=2,min_epochs=2,patience=1,batch_size=64,threads=1,normalization_method='min-max',
                deterministic=True,allow_tf32=False,grad_clip_norm=0),source=source,
            splits=dict(train=dict(paths=files[:1],physical_ids=['one'],nominal_capacity=1.1),
                        valid=dict(paths=files[1:2],physical_ids=['two'],nominal_capacity=1.1)),
            input_sha256={p:sha(p) for p in files[:2]},source_sha256={})
        records=[]
        for lane,freeze,mono,alpha in [('adapt','all_u','mean_eval',.1),('head','head_only','mean_eval',.1),
                ('target','all_u','mean_eval',0),('original','all_u','sum_train',.7)]:
            folder=temp/lane;folder.mkdir();w=copy.deepcopy(base);w['output']=str(folder);w['lane']=lane
            w['parameters'].update(freeze=freeze,mono=mono,alpha=alpha,beta=50 if mono=='sum_train' else .01 if alpha else 0)
            if lane=='target':w['source']=None;w['parameters']['initialization']='random'
            write(folder/'worker.json',w)
            assert files[2] not in str(w)
            command=[sys.executable,str(root/'reproduction/run_transfer_robustness.py'),'--worker-json',str(folder/'worker.json')]
            env=None
            import os
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
            run=subprocess.run(command,text=True,capture_output=True,encoding='utf-8',env=env)
            assert run.returncode==0,run.stderr
            r=read(folder/'result.json');assert r['G_frozen_unchanged'] and not r['test_loaded'] and r['epochs_completed']==2
            c=torch.load(folder/'best_model.pth',weights_only=True)
            assert state_hash(c['dynamical_F'])==(state_hash(g.state_dict()) if lane!='target' else state_hash(c['dynamical_F']))
            if freeze=='head_only':
                assert state_hash({k:v for k,v in c['solution_u'].items() if k.startswith('encoder.')})==state_hash({'encoder.'+k:v for k,v in u.encoder.state_dict().items()})
                assert r['initial_solution_sha256']!=r['final_solution_sha256']
            history=read(folder/'history.json')
            assert all(h['gradient_finite'] and not h['G_training'] and h['diagnostic_weighted_gradient_l2']['data']>0 for h in history)
            records.append(dict(lane=lane,epochs=2,G_frozen=True,checkpoint_reload_equal=True,test_not_loaded=True))
            print('PASS '+lane+': train/valid only, frozen G/encoder, finite gradients, checkpoint reload',flush=True)
            if lane=='adapt':
                for mode in ('source_test','test'):
                    evaluation=copy.deepcopy(w);ef=temp/mode;ef.mkdir()
                    evaluation.update(mode=mode,output=str(ef),splits=dict(test=dict(paths=files[2:],physical_ids=['three'],nominal_capacity=1.1)),
                        input_sha256={files[2]:sha(files[2])})
                    if mode=='test':evaluation.update(trained_checkpoint=str(folder/'best_model.pth'),trained_checkpoint_sha256=sha(folder/'best_model.pth'))
                    write(ef/'worker.json',evaluation)
                    p=subprocess.run([sys.executable,str(root/'reproduction/run_transfer_robustness.py'),'--worker-json',str(ef/'worker.json')],
                        text=True,capture_output=True,encoding='utf-8',env=env)
                    assert p.returncode==0,p.stderr
                    assert read(ef/'result.json')['test_loaded'] and read(ef/'result.json')['G_frozen_unchanged']
                print('PASS source-only and trained-checkpoint evaluation; test-only worker',flush=True)
        rows=[dict(candidate_id=c,seed=s,objective=v) for c,v in [('one',.2),('two',.1)] for s in [1,2,3]]
        cs=[dict(candidate_id='one',parameters={}),dict(candidate_id='two',parameters={})]
        assert freeze_selection(rows,cs,[1,2,3])['selected']['candidate_id']=='two'
        try:freeze_selection(rows[:-1],cs,[1,2,3]);raise AssertionError('Partial matrix accepted')
        except ValueError:pass
        print('PASS complete seed/candidate matrix and validation-only ranking',flush=True)
        print(identity(records))
    print('ALL_TRANSFER_ROBUSTNESS_FUNCTIONAL_TESTS_PASSED')


if __name__=='__main__':main()
