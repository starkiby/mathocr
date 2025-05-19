## Research on applying GRPO to Qwen2.5-VL-3B-Instruct

With inspiration from DeepSeek’s Reinforcement Fine-Tuning strategy, the Group Relative Policy Optimization (first introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300)), is a reinforcement learning–based fine-tuning method that compares batches of candidate outputs and updates the policy to favor higher-quality responses, this research tests applying GRPO to finetune [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).


### Motivation

The experiment was conducted on the [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) to enable the 3B model with a high capability for equation recognition and LaTeX generation. Although the current model, after SFT, can recognize equations well and generate LaTeX correctly, the output formats vary—for example, ` ```latex...``` `, `<latex>...</latex>`, `\[...\]`. Moreover, the output often contains unnecessary context, such as “Sure! The equation in the image can be written in LaTeX as follows…” or “That’s the generated LaTeX code. If you need an explanation of what the equations mean or how to use them, just tell me…”

The GRPO uses a rule-based reward function as the reward model for its policy updates, combining strict format matching with semantic equivalence checking to guide fine-tuning. Thus, we can design some reward functions to lead the model to generate the outputs as we required. Consequently, we designed 4 reward functions:

- **strict_format_reward_func**: checks if the completion has a specific format, with 0.5 reward. We require the model generate the latex strictly in ```<latex>{latex code}</latex>```.

- **stric_correct_reward_func**: checks the output latex is completely the same with the label. Since it's a strict one, the reward is 2.0.

- **soft_correct_reward_func**: Calculate the soft matching reward, if the parsed mathematical expressions are equivalent, return a score of 1.0; otherwise, return 0.0. In this function, we use Sympy’s LaTeX parser `parse_latex` to parse both the generated LaTeX and the reference answer into mathematical expressions.

- **length_penalty_reward**: Only apply length penalty (linear decay when exceeding the limit) to request the model's output as short as possible.


### Training

To accelerate the training time, we apply [ZeRO-2](src/grpo_ft/zero.yaml) for distribution training. The model configuration are detailed in [run.yaml](src/grpo_ft/run.yaml).

#### Test training

As the algorithm requires high GPU memory on single card, we only conducted a test training on 4 * 4090D for 0.3 epochs, the [training progresses](https://swanlab.cn/@Wonster/finetune/runs/ygggj56ppauts9um8vih8/chart) are as follows:

![](https://gitlab.cecs.anu.edu.au/u7729914/mathocr-25s1/-/raw/grpo_ft_research/src/grpo_ft/meta/test1.png)
![](https://gitlab.cecs.anu.edu.au/u7729914/mathocr-25s1/-/raw/grpo_ft_research/src/grpo_ft/meta/test2.png)

#### Next Step

- [ ] Training on HPC in CSIRO.