"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

#from model import DeepQNetwork, QTrainer
from game_ai import TetrisGameAI
from collections import deque
from deep_q_network import DeepQNetwork
import plotly.graph_objects as go


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = TetrisGameAI()
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    env.reset()
    state = env.get_state(env.board)

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    games = 0
    score_plot = []
    cleared_plot = []
    avg_score_plot = []
    avg_cleared_plot = []
    while epoch <= opt.num_epochs:
        next_steps = env.get_all_possible_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.play_step(action)

        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            games += 1

            score_plot.append(final_score)
            avg_score_plot.append(np.average(np.array(score_plot)))
            cleared_plot.append(final_cleared_lines)
            avg_cleared_plot.append(np.average(np.array(cleared_plot)))

            env.reset()
            state = env.get_state(env.board)
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Epsilon: {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            round(epsilon, 4),
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

        if epoch % 500 == 0:
            plot(games, epoch, score_plot, cleared_plot, avg_score_plot, avg_cleared_plot)

    torch.save(model, "{}/tetris".format(opt.saved_path))

def plot(games, epoch, scores, cleared, avg_score_plot, avg_cleared_plot):
    avg_score = np.average(np.array(scores))
    avg_cleared = np.average(np.array(cleared))
    x = list(range(1, games + 1))
    fig = go.Figure([
        go.Scatter(x=x, y=scores, mode='lines', name="Score", showlegend=True),
        go.Scatter(x=x, y=avg_score_plot, mode='lines', name="Average", showlegend=True)
    ]).update_layout(title=f"Avg score = {avg_score}, Avg cleared = {avg_cleared}")
    fig.write_image(f"images/score_{epoch}.png")

    fig = go.Figure([
        go.Scatter(x=x, y=cleared, mode='lines', name="Cleared", showlegend=True),
        go.Scatter(x=x, y=avg_cleared_plot, mode='lines', name="Average", showlegend=True)
    ]).update_layout(title=f"Avg score = {avg_score}, Avg cleared = {avg_cleared}")
    fig.write_image(f"images/cleared_{epoch}.png")

if __name__ == "__main__":
    opt = get_args()
    train(opt)
