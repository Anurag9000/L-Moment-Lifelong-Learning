"""Trainable NCD/GCD/C-GCD reference methodology on image streams.

This is an implementation of supervised-base training + unlabeled discovery,
pseudo-label consistency, stage-wise distillation and optionally distribution
shape preservation. It is **not** a reproduction of SimGCD, DYDM or any other
published architecture. Ground-truth novel-class labels are used only by the
benchmark protocol builder and evaluator, never in student optimizer targets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def _require_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
    return torch, nn, DataLoader, Dataset, Subset


@dataclass
class GCDConfig:
    known_classes: int = 5
    oracle_total_classes: int = 10
    mode: str = "gcd"  # ncd, gcd, continual_gcd
    epochs_base: int = 1
    epochs_session: int = 1
    batch_size: int = 32
    lr: float = .001
    seed: int = 0
    supervised_weight: float = 1.
    pseudo_weight: float = 1.
    consistency_weight: float = .1
    preservation_weight: float = .05

    def __post_init__(self):
        if self.mode not in ("ncd", "gcd", "continual_gcd"):
            raise ValueError("Choose NCD, GCD or C-GCD")
        if not (1 <= self.known_classes < self.oracle_total_classes <= 10):
            raise ValueError("Invalid known/novel class composition")
        if min(self.epochs_base, self.epochs_session, self.batch_size) < 1:
            raise ValueError("Need positive training epochs and batch size")


def get_labels(dataset):
    """Labels are accessible to the protocol scheduler/evaluator, not the learner."""
    _, _, _, _, Subset = _require_torch()
    if isinstance(dataset, Subset):
        return np.asarray(get_labels(dataset.dataset))[np.asarray(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    # Fallback to a dataset with a documented label-bearing contract.
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


class _ImagesWithoutLabels:
    """The unlabeled trainer never receives original category annotations."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][0]


class _PseudoTargetView:
    """Pseudo IDs come from clustering; source annotations are discarded."""
    def __init__(self, dataset, pseudo, weights):
        if len(dataset) != len(pseudo) or len(pseudo) != len(weights):
            raise ValueError("Pseudo assignment length mismatch")
        self.dataset = dataset
        self.pseudo = np.asarray(pseudo, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][0], int(self.pseudo[i]), float(self.weights[i])


def _features(encoder, image_dataset, device, batch_size):
    torch, nn, DataLoader, _, _ = _require_torch()
    encoder.eval()
    vals=[]
    with torch.inference_mode():
        for images in DataLoader(_ImagesWithoutLabels(image_dataset), batch_size=batch_size):
            z = nn.functional.normalize(encoder(images.to(device)), dim=1)
            vals.append(z.cpu().numpy())
    return np.concatenate(vals)


def _pseudo_assign(x, k, anchors, rng_seed, known, mode):
    if mode == "ncd":
        cluster = KMeans(n_clusters=k-known, n_init=5, random_state=rng_seed).fit(x)
        return cluster.labels_.astype(int) + known, cluster.cluster_centers_
    cluster=KMeans(n_clusters=k, n_init=5, random_state=rng_seed).fit(x)
    centroids=cluster.cluster_centers_
    old=np.stack([anchors[i] for i in range(known)])
    cost=np.square(old[:,None]-centroids[None]).sum(2)
    rr,cc=linear_sum_assignment(cost)
    mapping={int(c):int(r) for r,c in zip(rr,cc)}
    unused=[i for i in range(k) if i not in mapping]
    for j,c in enumerate(unused):mapping[c]=known+j
    pseudo=np.asarray([mapping[int(i)] for i in cluster.labels_])
    return pseudo,centroids


def _pseudo_reliability(features, pseudo, seed, statistic):
    if statistic == "lmoment":
        from .discovery import gcd_pseudolabel_weights
        return gcd_pseudolabel_weights(features, pseudo, np.ones(len(pseudo)),
                                      seed=seed, projections=min(features.shape[1],16),
                                      min_support=8)
    # Matched-confidence baseline without the group reliability statistic.
    # Normalize to a comparable mean so the loss weighting is controlled.
    return np.ones(len(features), dtype=np.float32)


def _distribution_consistency(student, teacher, variant):
    """Compare distributional statistics on the *same* current images."""
    torch, nn, _, _, _ = _require_torch()
    if len(student) < 4:
        return torch.zeros((), device=student.device)
    if variant == "lmoment":
        from .advanced_statistics import torch_lmoments
        return torch.square(torch_lmoments(student.float(), order=4) -
                            torch_lmoments(teacher.float(), order=4)).mean()
    # Ordinary moment control, no claim of matching exact computational cost.
    a = torch.stack((student.mean(0), student.std(0, unbiased=False)))
    b = torch.stack((teacher.mean(0), teacher.std(0, unbiased=False)))
    return torch.square(a-b).mean()


def _eval_stage(encoder, classifier, dataset, known, cutoff, device, batch_size):
    """Returns raw pseudo IDs and evaluator-only labels; does not optimize."""
    torch, nn, DataLoader, _, _ = _require_torch()
    encoder.eval();classifier.eval()
    predictions=[];labels=[]
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    with torch.inference_mode():
        for images,y in loader:
            logits=classifier(encoder(images.to(device)))
            pred=logits[:,:cutoff].argmax(-1).cpu().numpy()
            predictions.append(pred);labels.append(y.numpy())
    return np.concatenate(predictions),np.concatenate(labels)


def _validation_alignment(y, pred, cutoff):
    included=y<cutoff
    yy=np.asarray(y)[included];p=np.asarray(pred)[included]
    if not len(yy):return {}
    old,pidx=np.unique(p,return_inverse=True)
    truth,yidx=np.unique(yy,return_inverse=True)
    c=np.zeros((len(old),len(truth)),dtype=int)
    np.add.at(c,(pidx,yidx),1)
    r,col=linear_sum_assignment(c.max()-c)
    return {int(old[i]):int(truth[j]) for i,j in zip(r,col)}


def _stage_scores(y, pred, mapping, cutoff, known):
    mask=np.asarray(y)<cutoff
    truth=np.asarray(y)[mask]
    mapped=np.asarray([mapping.get(int(i),-1) for i in np.asarray(pred)[mask]])
    old=truth<known
    novel=~old
    return {"all_accuracy":float(np.mean(mapped==truth)),
            "old_accuracy":float(np.mean(mapped[old]==truth[old])) if old.any() else None,
            "novel_accuracy":float(np.mean(mapped[novel]==truth[novel])) if novel.any() else None,
            "seen_classes":int(cutoff),
            "evaluation_alignment": "fitted_on_separate_validation_partition"}


def train_gcd(encoder, dimension, train, validation, test, cfg, *, variant="baseline", device=None):
    """Train all stages from images and evaluate with validation-fixed mapping.

    The train/validation/test datasets must be disjoint. The scheduler is
    permitted to inspect labels for stage eligibility; no novel labels reach
    the student through the pseudo-target dataset.
    """
    torch, nn, DataLoader, _, Subset = _require_torch()
    if variant not in ("baseline","lmoment"):
        raise ValueError("Bad L-moment ablation choice")
    torch.manual_seed(cfg.seed)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    net=copy.deepcopy(encoder).to(device)
    classifier=nn.Linear(dimension,cfg.oracle_total_classes).to(device)
    optimizer=torch.optim.AdamW(list(net.parameters())+list(classifier.parameters()),lr=cfg.lr)
    y_train=get_labels(train)
    known_ids=np.flatnonzero(y_train<cfg.known_classes)
    known_dataset=Subset(train,known_ids)
    if not len(known_dataset):
        raise ValueError("Known-class training base absent")
    known_loader=DataLoader(known_dataset,batch_size=cfg.batch_size,shuffle=True)
    base_trace=[]
    for _ in range(cfg.epochs_base):
        net.train();classifier.train()
        for images,y in known_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(classifier(net(images.to(device))),y.to(device))
            loss.backward();optimizer.step()
            base_trace.append(float(loss.detach()))
    teacher=copy.deepcopy(net).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    # Teacher classifier is kept for historical logit preservation (base only).
    teacher_classifier=copy.deepcopy(classifier).eval()
    for p in teacher_classifier.parameters():p.requires_grad_(False)
    cutoffs = ([cfg.known_classes + max(1,(cfg.oracle_total_classes-cfg.known_classes)//2),
                cfg.oracle_total_classes] if cfg.mode == "continual_gcd"
               else [cfg.oracle_total_classes])
    session_records=[]
    for session,cutoff in enumerate(cutoffs):
        # Only the protocol builder inspects hidden novel-class labels.
        if cfg.mode=="ncd":
            session_ids=np.flatnonzero((y_train>=cfg.known_classes)&(y_train<cutoff))
        else:
            session_ids=np.flatnonzero(y_train<cutoff)
        current=Subset(train,session_ids)
        current_features=_features(net,current,device,cfg.batch_size)
        known_features=_features(net,known_dataset,device,cfg.batch_size)
        anchors={k:known_features[get_labels(known_dataset)==k].mean(0)
                 for k in range(cfg.known_classes)}
        pseudo,centers=_pseudo_assign(current_features,cutoff,anchors,
                                      cfg.seed+session,cfg.known_classes,cfg.mode)
        weights=_pseudo_reliability(current_features,pseudo,
                                     cfg.seed+session,variant)
        pseudo_loader=DataLoader(_PseudoTargetView(current,pseudo,weights),
                                 batch_size=cfg.batch_size,shuffle=True)
        stage_trace=[]
        for _ in range(cfg.epochs_session):
            known_iter=iter(known_loader)
            net.train();classifier.train()
            for images,targets,importance in pseudo_loader:
                try:old_images,old_y=next(known_iter)
                except StopIteration:
                    known_iter=iter(known_loader)
                    old_images,old_y=next(known_iter)
                images=images.to(device)
                old_images=old_images.to(device)
                old_y=old_y.to(device)
                targets=targets.to(device)
                importance=importance.to(device)
                opt_logits=classifier(net(images))
                pseudo_ce=nn.functional.cross_entropy(opt_logits,targets,reduction="none")
                pseudo_loss=(pseudo_ce*importance).sum()/importance.sum().clamp_min(1)
                labeled_loss=nn.functional.cross_entropy(classifier(net(old_images)),old_y)
                # A cheap augmentation-consistency term independent of labels.
                jitter=images+.025*torch.randn_like(images)
                student=net(images);augmented=net(jitter)
                consistency=nn.functional.mse_loss(student,augmented)
                with torch.no_grad():
                    historical=teacher(images)
                    old_logits=teacher_classifier(historical)
                preservation=_distribution_consistency(student,historical,variant)
                kd=nn.functional.mse_loss(opt_logits[:,:cfg.known_classes],
                                          old_logits[:,:cfg.known_classes])
                loss=(cfg.pseudo_weight*pseudo_loss+
                      cfg.supervised_weight*labeled_loss+
                      cfg.consistency_weight*consistency+
                      cfg.preservation_weight*(preservation+kd))
                optimizer.zero_grad(set_to_none=True)
                loss.backward();optimizer.step()
                stage_trace.append(float(loss.detach()))
        validation_pred,val_y=_eval_stage(net,classifier,validation,cfg.known_classes,
                                         cutoff,device,cfg.batch_size)
        test_pred,test_y=_eval_stage(net,classifier,test,cfg.known_classes,
                                    cutoff,device,cfg.batch_size)
        mapping=_validation_alignment(val_y,validation_pred,cutoff)
        scores=_stage_scores(test_y,test_pred,mapping,cutoff,cfg.known_classes)
        scores.update({"session":session+1,"train_examples":len(current),
                       "oracle_k":cutoff,"pseudo_groups":len(np.unique(pseudo)),
                       "optimizer_updates":len(stage_trace),
                       "final_training_loss":stage_trace[-1],
                       "validation":_stage_scores(val_y,validation_pred,mapping,
                                                  cutoff,cfg.known_classes)})
        session_records.append(scores)
        # Preserve previous-stage knowledge, not the original base forever.
        teacher=copy.deepcopy(net).eval()
        teacher_classifier=copy.deepcopy(classifier).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        for p in teacher_classifier.parameters():p.requires_grad_(False)
    return {"model_state":{"encoder":net.state_dict(),
                           "classifier":classifier.state_dict()},
            "results":{"mode":cfg.mode,"variant":variant,
                       "known_supervised_classes":cfg.known_classes,
                       "oracle_class_count":cfg.oracle_total_classes,
                       "novel_labels_in_optimizer":False,
                       "historical_base_labeled_replay":True,
                       "representation_trainable":True,
                       "base_optimizer_updates":len(base_trace),
                       "base_final_loss":base_trace[-1],
                       "sessions":session_records,
                       "status":"trainable_GCD_reference_not_SimGCD_or_DYDM"}}
