# Reuse out-of-year data to enhance land cover mapping via feature disentanglement and contrastive learning
  
The proposed approach corresponds to file `main_REFeD.py`. It employs a feature disentanglement procedure by extracting domain-invariant and domain-specific features using two independent encoders.
- The domain-invariant features are used for the final classification task.
- The domain-specific features are sent to a domain classifier, which tries to identify the provenance domain of the data (source or target).
- Disentanglement between the two types of encodings (domain invariant vs. specific) is enforced via a supervised contrastive loss which clusters together encodings of the same type and class, while repelling unsimilar ones i.e. with different provenance domain and/or class.

![alt text](./Arch.eps)

**Data** should be places in a folder (or soft link) or `./DATA_[dataset]/` (e.g. dataset = Koumbia or CVL3)

**Input arguments** Most scripts require the following input arguments in order (they are used for data loading and may be modified to meet your own dataset conventions):
1) source year (not required on target-only competitors)
2) split id (from 0 to 4, in our case, indicating different train/test splits of the data)
3) target year
4) dataset name (Koumbia or CVL3, in our case)

Exemple of command for running REFeD: `$ python main_REFeD.py 2020 0 2021 Koumbia`
