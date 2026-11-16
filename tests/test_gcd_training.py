"""Trainable image-space discovery contracts with causal stage visibility."""
import copy
import numpy as np
import pytest

from lmoment_lifelong.gcd_training import GCDConfig, train_gcd, get_labels, _validation_alignment


def fake_dataset(seed, samples_per_class=8):
    torch = pytest.importorskip('torch')
    from torch.utils.data import Dataset
    class FakeImages(Dataset):
        def __init__(self):
            self.targets=np.repeat(np.arange(10), samples_per_class).tolist()
            generator=torch.Generator().manual_seed(seed)
            self.features=torch.randn(10*samples_per_class,3,32,32,generator=generator)
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):return self.features[i],int(self.targets[i])
    return FakeImages()


@pytest.mark.parametrize('mode',['ncd','gcd','continual_gcd'])
def test_trainable_baseline_gcd_mode(mode):
    torch=pytest.importorskip('torch')
    from torch import nn
    torch.set_num_threads(2)
    train=fake_dataset(1,8);val=fake_dataset(2,8);test=fake_dataset(3,8)
    net=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU(),
                      nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,8))
    config=GCDConfig(mode=mode,epochs_base=1,epochs_session=1,batch_size=32,seed=0)
    before=copy.deepcopy(net.state_dict())
    r=train_gcd(net,8,train,val,test,config,variant='baseline',device='cpu')
    assert r['results']['novel_labels_in_optimizer'] is False
    assert r['results']['representation_trainable'] is True
    assert len(r['results']['sessions'])==(2 if mode=='continual_gcd' else 1)
    assert all('old_accuracy' in x and 'novel_accuracy' in x for x in r['results']['sessions'])
    assert all(torch.equal(before[k],net.state_dict()[k]) for k in before)


def test_validation_mapping_does_not_use_test_truth():
    y=np.array([0,0,1,1]);pred=np.array([7,7,4,4])
    mapping=_validation_alignment(y,pred,2)
    assert mapping=={4:1,7:0}
    assert get_labels(fake_dataset(4,8)).shape==(80,)
