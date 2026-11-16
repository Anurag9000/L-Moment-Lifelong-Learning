"""Generate explicit, noninflated inventory of 141 proposed experiment families.

Registry presence never implies implementation. CI checks exact identity/counts and
none of the 141 families is allowed to claim publication-level completion by default.
"""
from __future__ import annotations
import json
from pathlib import Path

FAMILIES = {
    "E01": [
        "Estimator verification against known distributions",
        "Distribution-shape discrimination at matched lower moments",
        "Matched-low-order-L-moment counterexamples",
        "ImageNet-or-CIFAR feature distribution profiling",
        "Difficult visually similar category pairs",
        "Dependence via projected L-moments and L-comoments",
        "Low-sample contamination and tail robustness",
        "Streaming sketch accuracy versus memory and throughput",
        "Sequential drift detection at fixed false-alarm rate",
        "Class-conditional predictive distribution fitting",
    ],
    "E02": [
        "Offline unsupervised unknown-class-count clustering",
        "Causal online predict-before-update discovery",
        "Hidden-boundary class-blocked stream",
        "Blurry class arrival and mixed-class streams",
        "Historical class recurrence without duplicate identity",
        "Growing and unknown class counts",
        "Strict fixed-byte memory discovery",
        "Frozen-versus-trainable representation sensitivity",
    ],
    "E03": [
        "NCD with labeled known classes and novel-only unlabeled pool",
        "GCD with mixed known and novel unlabeled pool",
        "Oracle-K versus estimated-K GCD",
        "Frozen backbone versus trainable representation GCD",
        "Pseudo-label noise injection and statistical correction",
        "Old-new class frequency imbalance",
        "Visually similar fine-grained new classes",
        "Large-scale generalized category discovery",
    ],
    "E04": [
        "Continual generalized category discovery reference",
        "Frozen visual representation C-GCD",
        "Trainable visual representation C-GCD",
        "Long-horizon incremental session count",
        "Old versus new example imbalance",
        "Exemplar-free historical preservation",
        "Bounded exemplar C-GCD",
        "Historical pseudo-label contamination",
        "Historical novel-category recurrence",
        "Large-scale continual generalized category discovery",
    ],
    "E05": [
        "Online probabilistic versus threshold versus deferred class birth",
        "Class-birth detection delay as support grows",
        "Unknown and changing semantic class count",
        "Abrupt versus gradual arrival and false alarms",
        "Long-tailed rare novel class arrival",
        "Recurrent old categories and duplicate birth prevention",
        "Known-class frequency shift without semantic novelty",
        "Frozen versus causally adaptive encoder",
        "Fixed total memory under novel-class arrival",
        "Long-horizon on-the-fly discovery at large scale",
    ],
    "E06": [
        "Incorrect early fragmentation repaired by merging",
        "Incorrect early grouping repaired by splitting",
        "Multimodal appearance variation without false semantics",
        "Delayed structural revision from additional evidence",
        "Pseudo-label contamination and repair",
        "Old-category reactivation without identity duplication",
        "Domain submodes without incorrect semantic birth",
        "Rare class consolidation and postponement",
        "Protected memory reconsolidation",
        "History budget and eviction without silent loss",
        "Geometry versus covariance versus L-moment structural evidence",
        "Long-horizon structural memory and class count",
    ],
    "E07": [
        "Known semantic class under new image corruption",
        "Known class under new image style",
        "Novel class under a familiar domain",
        "Novel class simultaneous with new appearance domain",
        "Rare historical class recurrence",
        "Rare genuinely novel class detection",
        "Contamination by unrelated observations",
        "Gradual corruption severity evolution",
        "Appearance-domain recurrence and recovery",
        "Combined novelty domain noise imbalance and recurrence",
    ],
    "E08": [
        "Flat versus learned hierarchical category memory",
        "Coarse-to-fine discovered subcategory structure",
        "Fine-to-coarse parent discovery",
        "Closely related visual category discrimination",
        "Global versus local patch-token statistics",
        "Semantic children versus appearance submodes",
        "Fine-grained historical class recurrence",
        "Unknown number of subcategories",
        "Large-scale taxonomy reconstruction",
    ],
    "E09": [
        "Frozen-encoder Class-IL classifier comparison",
        "Large-scale exemplar-free Class-IL",
        "Incremental low-shot historical class fitting",
        "Long-tailed exemplar-free retention",
        "Similar-category higher-order distributional modeling",
        "Projection-count versus storage versus performance",
        "Parametric distribution-family misspecification",
        "Equal-byte exemplar-free statistical memory",
        "Trainable-encoder exemplar-free continual adaptation",
    ],
    "E10": [
        "Replay-policy comparison at equal exemplars per class",
        "Fixed-total-byte image versus feature memory",
        "Replay coverage of multimodal visual distributions",
        "Class-imbalanced replay selection",
        "Gradient interference-aware replay",
        "Replay ratio at equal optimization budget",
        "Changing feature encoder and stale memory",
        "Incorrect replay pseudo-label injection",
        "Recurrent concept reconsolidation",
        "Corruption versus actual concept change",
        "Long-horizon large-scale replay selection",
    ],
    "E11": [
        "Abrupt hidden-boundary class transition",
        "Gradual mixture drift without boundaries",
        "Domain-Incremental shift without new classes",
        "Recurrent historical domain and module reuse",
        "Harmless distribution change should not adapt",
        "Loss spike without environmental drift",
        "Distribution drift without loss deterioration",
        "Known task boundaries versus fully hidden boundaries",
        "Supervised task-free versus unlabeled SSL access",
        "Large-scale mixed nonstationary streams",
    ],
    "E12": [
        "Fixed architecture versus growing-width MLP",
        "Width increment and candidate parameter budget",
        "Loss-only versus L-moment-only growth criteria",
        "Joint drift loss and capacity-evidence controller",
        "Candidate growth trials versus untested expansion",
        "Task-ID-oracle neural capacity growth",
        "Single-head class-incremental neural growth",
        "No-boundary task-free capacity adaptation",
        "Recurrent historical task module reuse",
        "BGD posterior updates versus ordinary optimizer",
        "Predictive-uncertainty shape versus variance triggers",
        "Large-scale vision adapter and FFN-only growth",
    ],
    "E13": [
        "Online continual SSL on CIFAR-100",
        "Online continual SSL on Tiny-ImageNet",
        "Causal ImageNet-100 unlabeled single-pass learning",
        "Blurry nonstationary continual SSL",
        "Returning categories and historical representation retention",
        "Teacher-student historical feature distillation",
        "L-moment distributional SSL constraint",
        "Engram-retrieved self-supervised positive relations",
        "From-scratch versus pretrained continual SSL",
        "Large-scale continual SSL transfer and forgetting",
    ],
    "E14": [
        "Unified memory stability on known classes",
        "Unified autonomous novel-class formation",
        "Known-class appearance shift without false birth",
        "Historical class recurrence and memory reactivation",
        "Deliberately induced forgetting diagnosis and recovery",
        "Incorrect historical split-merge repair",
        "Verified architecture capacity shortage and appropriate growth",
        "New categories separable with existing capacity",
        "Rare classes outliers and budgeted memory",
        "Simultaneous competing sources of change",
        "Long-horizon integrated ImageNet-100",
        "Integrated ImageNet-1K scaling and Pareto comparison",
    ],
}

COUNTS = [10, 8, 8, 10, 10, 12, 10, 9, 9, 11, 10, 12, 10, 12]
assert [len(v) for v in FAMILIES.values()] == COUNTS
assert sum(COUNTS) == 141

# Only a handful of mechanisms currently have an independently runnable
# additional reference program; this list does not mean the family is complete.
REFERENCE = {"E01-D", "E02-B", "E03-B", "E04-A", "E05-A", "E06-B",
             "E07-G", "E08-A", "E09-A", "E10-A", "E11-A", "E12-A",
             "E13-C", "E14-B"}
CORE = {"E01", "E02", "E05", "E06", "E08", "E09", "E10", "E11"}


def registry():
    families = []
    for track, names in FAMILIES.items():
        for i, title in enumerate(names, 1):
            identifier = f"{track}-{chr(64 + i)}"
            families.append({
                "family_id": identifier,
                "track": track,
                "title": title,
                "dataset": "CIFAR-10 pilot when semantically applicable; full study may need other datasets",
                "scientific_implementation": "not_complete",
                "reference_available": identifier in REFERENCE,
                "extended_algorithm_api": track in CORE,
                "results": "not_verified",
                "published_baseline_reproduction": False,
                "required_evidence": ["algorithm and data contract", "baseline implementation",
                                      "matched ablation", "training or estimation run",
                                      "independent validation", "multi-seed comparison",
                                      "resource accounting"],
            })
    return {"schema": 1, "total_families": len(families),
            "method_scope": "planned research; registry presence does not imply implementation",
            "families": families}


def main():
    path = Path(__file__).with_name("research_families.json")
    path.write_text(json.dumps(registry(), indent=2) + "\n")
    print(f"Wrote {len(registry()['families'])} planned families to {path}")


if __name__ == "__main__":
    main()
