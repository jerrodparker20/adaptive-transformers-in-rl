# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
import pprint
import threading
import time
import timeit
import traceback
import typing
from StableTransformersReplication.transformer_xl import MemTransformerLM
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.


import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from Model.core import environment, file_writer, prof, vtrace
from Model import atari_wrappers


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="./logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")

# This is by default true in our case
# parser.add_argument("--use_lstm", action="store_true",
#                     help="Use LSTM in agent model.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--weight_decay", default=0.0,
                    type=float)
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="momentum for SGD or RMSProp")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument('--optim', default='RMSProp', type=str,
                    choices=['adam', 'sgd', 'adagrad, RMSProp'],
                    help='optimizer to use.')
parser.add_argument('--scheduler', default='torchLR', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'torchLR'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')

parser.add_argument('--use_pretrained', action='store_true',
                    help='use the pretrained model identified by --xpid')

# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        # explicitly make done False to allow the loop to run
        env_output['done'] = torch.tensor([0], dtype=torch.uint8)

        agent_state = model.initial_state(batch_size=1)

        # TODO DEBUG : negative probability coming up here
        # print('Env output shape 1: ',env_output['frame'].shape)
        # print('AGENT STATE: ', agent_state)

        agent_output, unused_state, mems = model(env_output, agent_state, mems=None)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do one new rollout, untill flags.unroll_length
            t = 0
            while t < flags.unroll_length and not env_output['done'].item():
            # for t in range(flags.unroll_length):
                timings.reset()

                if env_output['done'].item():
                    mems = None
                with torch.no_grad():
                    #HERE IS WHY B=1, T=1
                    # print('Env output shape: ',env_output['frame'].shape)
                    # print('ACTING')
                    agent_output, agent_state, mems = model(env_output, agent_state, mems)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
                t += 1

            if t != flags.unroll_length:
                # this means the episode ended before the rollout length
                # copy the buffer positions towards the end
                for key in env_output:
                    buffers[key][index][flags.unroll_length - t:] = buffers[key][index][:t+1]
                for key in agent_output:
                    buffers[key][index][flags.unroll_length - t:] = buffers[key][index][:t+1]
                # update the dones with True for beginning positions
                buffers['done'][index][:t+1] = torch.tensor([True]).repeat(t+1)

            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        # print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        """
        put a lock on the central learner,
        send the trajectories to it.
        Update the parameters of the central learner,
        copy the parameters of the central learner back to the actors
        """
        # print('RUNNING MAIN MODEL')
        # print('MODEL OUTOUT: ', model(batch, initial_agent_state))
        # print('batch size : ', batch['frame'].size())

        # TODO Next Step: think about chopping the sequences up for attending to long sequences
        # TODO Next Step: make the actors put only one sequence in one rollout. And then mask the remaining rollout
        #       positions. And then change the triu in line 461 in TXL
        # Here entire 81 length sequence is considered as a query, and is autoregressively being attended to the
        # keys and values of length 81. This sequence length when becomes very large is when we'll need memory to
        # kick in
        learner_outputs, unused_state, unused_mems = model(batch, initial_agent_state, mems=None)

        # Take final value function slice for bootstrapping.
        # this is the final value from this trajectory
        bootstrap_value = learner_outputs["baseline"][-1]


        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        # TODO Next Step: the losses also have to be computed with the padding, think on a structure of mask
        #                   to do this efficiently
        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        if flags.fp16:
            optimizer.clip_master_grads(flags.grad_norm_clipping)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        # scheduler is being stepped in the lock of batch_and_learn itself
        # scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def get_optimizer(flags, parameters):
    optimizer = None
    if flags.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=flags.learning_rate, momentum=flags.momentum)
    elif flags.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=flags.learning_rate)
    elif flags.optim.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(parameters, lr=flags.learning_rate)
    return optimizer


def get_scheduler(flags, optimizer):
    scheduler = None
    if flags.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         flags.max_step, eta_min=flags.eta_min)
    elif flags.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and flags.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > flags.warmup_step \
                    else step / (flags.warmup_step ** 1.5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif flags.scheduler == 'dev_perf':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=flags.decay_rate, patience=flags.patience,
                                                         min_lr=flags.lr_min)
    elif flags.scheduler == 'constant':
        pass
    return scheduler


def train(flags):  # pylint: disable=too-many-branches, too-many-statements

    # load the previous config if use_pretrained is true
    if flags.use_pretrained:
        logging.info('Using Pretrained Model')
        class Bunch(object):
            def __init__(self, adict):
                self.__dict__.update(adict)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs/torchbeast/'+flags.xpid+'/model.tar')
        pretrained_model = torch.load(model_path, map_location='cpu' if flags.disable_cuda else 'gpu')
        flags = Bunch(pretrained_model['flags'])
        flags.use_pretrained = True

    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    """model is each of the actors, running parallel. The upcoming block ctx.Process(...)"""
    model = Net(env.observation_space.shape, env.action_space.n)
    buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    """learner_model is the central learner, which takes in the experiences and updates itself"""
    learner_model = Net(
        env.observation_space.shape, env.action_space.n).to(device=flags.device)

    optimizer = get_optimizer(flags, learner_model.parameters())
    if optimizer is None:
        # Use the default optimizer used in monobeast
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
            weight_decay=flags.weight_decay
        )

    try:
        from apex.fp16_utils import FP16_Optimizer
    except:
        print('WARNING: apex not installed, ignoring --fp16 option')
        flags.fp16 = False

    if not flags.disable_cuda and flags.fp16:
        # If args.dynamic_loss_scale is False, static_loss_scale will be used.
        # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=flags.static_loss_scale,
                                   dynamic_loss_scale=flags.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16})

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = get_scheduler(flags, optimizer)
    if scheduler is None:
        # use the default scheduler as used in monobeast
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    if flags.use_pretrained:
        logging.info('Using Pretrained Model -> loading learner_model, optimizer, scheduler states')
        learner_model.load_state_dict(pretrained_model['model_state_dict'])
        optimizer.load_state_dict(pretrained_model['optimizer_state_dict'])
        scheduler.load_state_dict(pretrained_model['scheduler_state_dict'])

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            # print('Before Learn')
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            # print('After Learn')
            timings.time("learn")
            with lock:
                # step-wise learning rate annealing
                # TODO : How to perform annealing here exactly, we dont have access to the train_step !
                if flags.scheduler in ['cosine', 'constant', 'dev_perf']:
                    # linear warmup stage
                    if step < flags.warmup_step:
                        curr_lr = flags.lr * step / flags.warmup_step
                        optimizer.param_groups[0]['lr'] = curr_lr
                    else:
                        if flags.scheduler == 'cosine':
                            scheduler.step()
                elif flags.scheduler == 'inv_sqrt':
                    scheduler.step()

                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                # print('updating step from {} to {}'.format(step, step+(T*B)))
                step += T * B


        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            # TODO : We also should save the model if the loss is the best loss seen so far
            # TODO : call checkpoint() here with some differen prefix
            # if not best_val_loss or val_loss < best_val_loss:
            #     if not args.debug:
            #         with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
            #             torch.save(model, f)
            #         with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
            #             torch.save(optimizer.state_dict(), f)
            #     best_val_loss = val_loss

            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []

    mems = None
    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()

        # TODO: Check that this call to model is correct
        agent_outputs, core_state, mems = model(observation, mems=mems)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )



def init_weight(weight):
    #if args.init == 'uniform':
    #    nn.init.uniform_(weight, -args.init_range, args.init_range)
    #elif args.init == 'normal':
    nn.init.normal_(weight, 0.0, 0.02)#args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, 0.01)#args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, 0.01)#args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.02)#args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        # print('FOUND TRNASFORMER LM')
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)



class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
    
        self.convs = []
        input_channels=self.observation_shape[0]
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
                               
            input_channels = num_ch
                               
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                        nn.Conv2d(
                            in_channels=input_channels,
                            out_channels=num_ch,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                        nn.Conv2d(
                            in_channels=input_channels,
                            out_channels=num_ch,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
    
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)


        # Fully connected layer.
        # Changed the FC output to match the transformer output which should be divisible by number of heads
        self.fc = nn.Linear(3872, 256-num_actions-1)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1
        ###############################################################transformer
        # TODO : 1st replacement, sanity check the parameters
        # TODO : play around with d_inner, this is the dimension for positionwise feedforward hidden projection
        # TODO : Change the n_layer=1 to 12
        self.core = MemTransformerLM(n_token=None, n_layer=1, n_head=8, d_head=core_output_size // 8,
                                     d_model=core_output_size, d_inner=2048,
                                    dropout=0.1, dropatt=0.0, tgt_len=512, mem_len=1, ext_len=0,
                                     use_stable_version=True, use_gate=False)
        self.core.apply(weights_init)
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        # if not self.use_lstm:
        #     return tuple()
        return tuple(
            # torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            torch.zeros(self.core.n_layer, batch_size, self.core.d_model)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=(), mems=None):

        x = inputs["frame"]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        core_input = core_input.view(T, B, -1)

        #Need padding_mask to be qlenX(qlen+mlen)Xbatch_size and shouldn't have final
        #column masked.
        # TODO: This doesn't work if need to mask memory as well (right now we don't need to)

        padding_mask = inputs['done'].unsqueeze(0)
        if padding_mask.dim() > 2: #This only seems to not happen on first state ever in env.initialize()
            #print('PADDING DIM:', padding_mask.shape)
            padding_mask[:, 0, :] = True #TODO REMOVE THIS LINE
            padding_mask[:,-1,:] = False #if last row is Done then don't want to mask
            #print('ALL IS FALSE: {}, shape: {}'.format(padding_mask.any().item(), padding_mask.shape ))
        if not padding_mask.any().item(): #In this case no need for padding_mask
            padding_mask = None

        core_output, mems = self.core(core_input, mems, padding_mask=padding_mask)   # core_input is of shape (T, B, ...)
                                              # core_output is (B, ...)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        policy_logits = policy_logits.reshape(T*B, self.num_actions)
        if self.training:
            # Sample from multinomial distribution for exploration
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)

        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state, mems
        )


Net = AtariNet


def create_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)


# TODO : Should we have this functinality as well, to load the model if this is not a restart?
# if args.restart:
#     if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
#         with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
#             opt_state_dict = torch.load(f)
#             optimizer.load_state_dict(opt_state_dict)
#     else:
#         print('Optimizer was not saved. Start from scratch.')