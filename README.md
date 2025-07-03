# Generalizing Verifiable Instruction Following

This repo contains IFBench, which is a new, challenging benchmark for precise instruction following.

<p align="center"> <a href="https://github.com/allenai/IFBench/blob/main/Precise_IF_Generalization_Abilities.pdf"> Read the IFBench paper</a> </p>

Top open-source and commercial models that score > 80% on IFEval fall to roughly 50% or lower on IFBench, exposing significant room for AI progress.
![image](https://github.com/user-attachments/assets/80e77314-2ef6-4403-a49d-545759f49fbc)




## IFBench
IFBench consists of two parts:

- OOD Constraints: 58 new and challenging constraints, with corresponding verification functions. The constraint templates are combined with prompts from a held-out set of WildChat (Zhao et al. 2024).

- (optionally) Multiturn Constraint Isolation in 2 turns: The prompt and the constraint are separated over two turns, i.e. the first turn is the user prompt and the model's response to the prompt, and the second turn is the constraint that modifies the initial prompt.

- New IF-RLVR training constraints: 29 new and challenging constraints, with corresponding verification functions. 

## How to run the evaluation
Install the requirements via the requirements.txt file.
You need two jsonl files, one the IFBench_test.jsonl file (in the data folder) and one your file with eval prompts and completions (see sample_output.jsonl as an example). Then run:
```
python3 -m run_eval --input_data=IFBench_test.jsonl --input_response_data=sample_output.jsonl --output_dir=eval
```

## Released Datasets
You can find our released datasets in this [collection](https://huggingface.co/collections/allenai/ifbench-683f590687f61b512558cdf1), which contains the [test data](https://huggingface.co/datasets/allenai/IFBench_test), the [multi-turn test data](https://huggingface.co/datasets/allenai/IFBench_multi-turn) and the [IF-RLVR training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5).

## RLVR for Precise Instruction Following
We also release our IF-RLVR code, as part of [open-instruct](https://github.com/allenai/open-instruct). You can run this [GRPO script](https://github.com/allenai/open-instruct/blob/main/open_instruct/grpo_fast.py), using our [training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5). This is an [example command](https://github.com/allenai/open-instruct/blob/main/scripts/train/rlvr/valpy_if_grpo_fast.sh).

The new training constraints and verification functions are here: https://github.com/allenai/open-instruct/tree/main/open_instruct/IFEvalG

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The data is licensed under ODC-BY-1.0. It is intended for research and educational use in accordance with Ai2's Responsible Use Guidelines. The dataset includes output data generated from third party models that are subject to separate terms governing their use.


## Acknowledgements

Parts of IFBench are built upon and extend [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) (Zhou et al. 2023) and we would like to thank them for their great work!


## Citation

If you used this repository or our models, please cite our work:

```bibtex
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following}, 
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
   eprint={TODO},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
