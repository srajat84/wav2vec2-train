
Code base for IITM English ASR Challenge

# Experiment 2 Details

## Folder structure explanation:
Whenever you are starting with a new language make sure to create a new branch in this repository.
All executed scripts must be pushed to this repo by **creating PULL Request** only.
Only after a pull request is approved and merged, we will proceed with training.

```
checkpoints : will contain the trained checkpoints. Since there is a limit to push only till 25 MB file size in git this folder is by default put in gitignore. The checkpoints after getting trained should be uploaded to GCP bucket.

data: this will contain the data files used for finetuning or pretraining. tsv's , wrd , lexicon and dict files only.
```

| Type | Value           |
|------------------|-------------------|
| Pretraining Model | Base             |
| Pretraining Done | Yes                |
| Pretraining Model path on bucket | path               |
| Pretraining completed in | time |
| Pretraining updates run | 10000 |
| Pretraining log file path | path |
| Finetuning Done  | No               |
| Finetuning Data  | 280 hr IITM-NPTEL |
| Finetuning Model path on bucket | path |
| Finetuning completed in | time |
| Finetuning updates run | 10000 |
| Finetuning log file path | path |
| Language Model Used | No |
| Language Model Type | Kenlm |
| Language Model Training Data | Training Data |
| Beam Width Used | 128 |
| Inference results path | path |
| Inference done on set | Set name |
| WER | 10 |
| CER | 10 |
