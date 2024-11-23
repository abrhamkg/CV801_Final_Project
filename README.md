# Show and Tell: A Language-Modeling Framework for Referring Expression Counting

This is the official repository for the Show, Count and Tell (SCT): paper available at: https://arxiv.org/pdf/2402.10698.pdf

## Model Description
SCT Show Count and Tell” (SCT) an architecture that tackles referring expression counting as a text-to-text approach, with the main goal of taking advantage of the powerful reasoning capabilities of current LLMs and simplifying previous methods in this task

<p align="center">
  <img width="700" alt="fig1" src="https://github.com/abrhamkg/CV801_Final_Project/blob/b2630d1ff11e086923350c9e2bead97deeedc52f/ar.png">
</p>

SCT relies on [Paligemma](https://huggingface.co/docs/transformers/main/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration), we use the original pretrained checkpoint 0n 224x224 resolution [PaliGemma-PT-224](https://huggingface.co/google/paligemma-3b-pt-224). Our trained model can be downloaded from:

| Hugging Face Model  |
| ------------- |
| [SCT-PaliGemma-224](Daromog/paligemma-cord-demo-rand-50epo)
