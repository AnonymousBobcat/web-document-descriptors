# web-document-descriptors
Repository for code used in web document descriptor paper.

Repository for code used in web document descriptor paper.

To replicate the pipeline in the paper.

1. Install dependencies in requirement.txt
1. Generate descriptors with descriptor_generation/generate_descriptors.py
2. For disambiguation, first group definitions and tags with disambiguation/extract_descriptor_groups.py. The output of this should then be used as input for disambiguation/disambiguate_descriptors.py. Repeat the extraction-disambiguation process until no or few duplicates remain.
3. Merge synonyms with merging/merge_synonyms.py. If any tag duplicates still remain, run merging/force_merge.py to merge them.
4. For evaluation, use the faiss/search.py to find descriptors by query and the LLM_as_judge/LLM_as_judge.py to evaluate the results. To train a light-weight classifier, run classification/2_classify.py.
