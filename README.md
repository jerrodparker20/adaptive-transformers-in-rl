# Adaptive Transformers  in RL

Official implementation of [Adaptive Transformers in RL](http://arxiv.org/abs/2004.03761)

In this work we replicate several results from [Stabilizing Transformers for RL](https://arxiv.org/abs/1910.06764) on both [Pong](https://gym.openai.com/envs/Pong-v0/) and [rooms_select_nonmatching_object](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30#select-non-matching-object) from DMLab30. 

We also extend the Stable Transformer architecture with [Adaptive Attention Span](https://arxiv.org/abs/1905.07799) on a partially observable (POMDP) setting of Reinforcement Learning. To our knowledge this is one of the first attempts to stabilize and explore Adaptive Attention Span in an RL domain.

### Steps to replicate what we did on your own machine
1. Downloading DMLab: 
    * Build DMLab package with Bazel– https://github.com/deepmind/lab/blob/master/docs/users/build.md
    * Install the python module for DMLab– https://github.com/deepmind/lab/tree/master/python/pip_package

2. Downloading Atari: Getting Started with Gym– http://gym.openai.com/docs/#getting-started-with-gym

3. Execution notes:
* The experiments take around 4 hours on 32vCPUs and 2 P100 GPUs for 6 million environment interactions.
To run without a GPU, use the flag “--disable_cuda”.
* For more details on other flags, see the top of train.py (include a link to this file) which has descriptions for each.
* All experiments use a slightly revised version of [IMPALA](https://arxiv.org/abs/1802.01561) from [torchbeast](https://github.com/facebookresearch/torchbeast)

### Snippets
Best performing adaptive attention span model on “rooms_select_nonmatching_object”:
```
python train.py --total_steps 20000000 \
--learning_rate 0.0001 --unroll_length 299 --num_buffers 40 --n_layer 3 \
--d_inner 1024 --xpid row85 --chunk_size 100 --action_repeat 1 \
--num_actors 32 --num_learner_threads 1 --sleep_length 20 \
--level_name rooms_select_nonmatching_object --use_adaptive \
--attn_span 400 --adapt_span_loss 0.025 --adapt_span_cache
```

Best performing Stable Transformer on Pong:
```
python train.py --total_steps 10000000 \
--learning_rate 0.0004 --unroll_length 239 --num_buffers 40 \
--n_layer 3 --d_inner 1024 --xpid row82 --chunk_size 80 \
--action_repeat 1 --num_actors 32 --num_learner_threads 1 \
--sleep_length 5 --atari True
```

Best performing Stable Transformer on “rooms_select_nonmatching_object”:
```
python train.py --total_steps 20000000 \
--learning_rate 0.0001 --unroll_length 299 \
--num_buffers 40 --n_layer 3 --d_inner 1024 \
--xpid row79 --chunk_size 100 --action_repeat 1 \
--num_actors 32 --num_learner_threads 1 --sleep_length 20 \
--level_name rooms_select_nonmatching_object  --mem_len 200
```

### Reference
If you find this repository useful, do cite it with,
```
@article{kumar2020adaptive,
    title={Adaptive Transformers in RL},
    author={Shakti Kumar and Jerrod Parker and Panteha Naderian},
    year={2020},
    eprint={2004.03761},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
