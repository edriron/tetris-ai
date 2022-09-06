from settings import *
import torch
import random
from model import QTrainer, DeepQNetwork
# from deep_q_network import DeepQNetwork
from collections import deque
from game_ai import TetrisGameAI
import numpy as np


class Agent:

    def __init__(self, gamma=0.99, epsilon_range=(1, 0.01), epsilon_decay_rate=4000,
                 final_epoch=60000, max_memory=30000, batch_size=512, save_intervals=500,
                 input_size=4, hidden_size=256, output_size=1, hidden_layers=1):
        self.games_count = 0
        self.starting_epsilon = epsilon_range[0]
        self.final_epsilon = epsilon_range[1]
        self.epsilon = self.starting_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.epoch = 0
        self.final_epoch = final_epoch
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.save_intervals = save_intervals
        self.memory = deque(maxlen=self.max_memory)
        self.model = DeepQNetwork(input_size, hidden_size, output_size, hidden_layers)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, reward, next_state, game_over):
        self.memory.append([state, reward, next_state, game_over])

    def train_short_memory(self, next_states):
        return self.trainer.train_step(next_states)

    def train_long_memory(self):
        self.epoch += 1
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # sample randomly from queue
        else:
            mini_sample = self.memory

        state_s, reward_s, next_state_s, game_over_s = zip(*mini_sample)
        state_s = torch.stack(tuple(state for state in state_s))
        reward_s = torch.from_numpy(np.array(reward_s, dtype=np.float32)[:, None])
        next_state_s = torch.stack(tuple(state for state in next_state_s))

        if torch.cuda.is_available():
            state_s = state_s.cuda()
            reward_s = reward_s.cuda()
            next_state_s = next_state_s.cuda()

        q_values = self.model(state_s)
        next_preds = self.trainer.train_step(next_state_s, slice=False)

        y_batch = torch.cat(
            tuple(reward if game_over else reward + self.gamma * pred
                  for reward, game_over, pred in zip(reward_s, game_over_s, next_preds)))[:, None]
        self.trainer.optimizer.zero_grad()
        loss = self.trainer.loss_func(q_values, y_batch)
        loss.backward()
        self.trainer.optimizer.step()

    def calc_epsilon(self):
        return self.final_epsilon + (max(self.epsilon_decay_rate - self.epoch, 0) *
                                     (self.starting_epsilon - self.final_epsilon) / self.epsilon_decay_rate)

    def get_action(self, next_steps, next_states, next_actions, preds):
        # self.epsilon = RANDOM_GAMES_BATCH - self.games_count
        self.epsilon = self.calc_epsilon()

        if random.random() <= self.epsilon:
            # if random.randint(0, RANDOM_GAMES_BATCH * 2.5) < self.epsilon or random.random() <= 0.01:
            index = random.randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(preds).item()

        return next_states[index, :], next_actions[index]


def train():
    record = 0
    agent = Agent()
    game = TetrisGameAI()

    # current state
    state = game.get_state(game.board)
    if torch.cuda.is_available():
        agent.model.cuda()
        state = state.cuda()

    while agent.epoch < agent.final_epoch:
        # all (possible) states for next move
        next_steps = game.get_all_possible_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # train
        preds = agent.train_short_memory(next_states)

        # get next action and its state - exploration or exploitation stage
        next_state, action = agent.get_action(next_steps, next_states, next_actions, preds)

        # play the chosen action
        reward, game_over = game.play_step(action)
        if torch.cuda.is_available():
            next_state = next_state.cuda()

        # add to memory
        agent.remember(state, reward, next_state, game_over)

        if game_over:
            # train long and plot result
            cleared = game.cleared_lines
            tetrominoes = game.tetrominoes
            final_score = game.score
            game.reset()
            state = game.get_state(game.board)
            if torch.cuda.is_available():
                state = state.cuda()
            agent.games_count += 1

            print(f"Game: {agent.games_count}, Epoch: {agent.epoch}/{agent.final_epoch},"
                  f" Cleared lines: {cleared}, Record: {record},"
                  f" Tetrominoes: {tetrominoes}, Score: {final_score}"
                  f" Epsilon: {round(agent.epsilon, 4)}")

            # don't train on whole memory when not enough
            if len(agent.memory) >= agent.max_memory / 10:
                agent.train_long_memory()

            if cleared > record:
                record = cleared

            if agent.epoch % agent.save_intervals == 0:
                agent.model.save()

        else:
            state = next_state


if __name__ == '__main__':
    train()
