# Reuse out-of-year data to enhance land cover mapping via feature disentanglement and contrastive learning
Contrastive supervised domain adaptation applied to Land Cover mapping

The proposed approach corresponds to file main_combined_source_target_dis_V4.py, which employs a feature disentanglement procedure by extracting domain_invariant and domain-specific featureswith two independant encoders. The domain-invariant features are used for the final classification task. The domain-specific features are sent to a domain classifier, which tries to identify the provenance domain of the data (source or target). Disentanglement between these two types of encodings (domain invariant vs. specific) is enforced via a supervised contrastive loss which clusters together encodings of the same type and class, while repelling unsimilar ones (different provenance domain and/or class). 

Data should be places in a folder (or soft link) 'DATA/' or 'DATA_Koumbia/' and 'DATA_CVL3/' in the project root.
