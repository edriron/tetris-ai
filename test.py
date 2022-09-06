import game_ai as g1
import tetris as g2
import random
import torch
from copy import deepcopy
from model import DeepQNetwork
import pytest
import numpy

class Test:
    def __init__(self, width=600, height=1200, block_size=20):
        self.w = width
        self.h = height
        self.block_size = block_size
        self.width = int(self.w / self.block_size)
        self.height = int(self.h / self.block_size)
        self.g1 = g1.TetrisGameAI(h=self.h, w=self.w, reset=False)
        self.g2 = g2.Tetris(height=self.height, width=self.width, reset=False)

    def random_board(self, epsilon=0.5, lines=False):
        board = [[0] * self.width for i in range(self.height)]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if epsilon <= random.random():
                    board[i][j] = 1
                if lines and random.randint(0, 10) == 10:
                    board[i][j] = 1
        return board

    def test_holes(self):
        board_empty = [[0] * self.width for i in range(self.height)]
        assert self.g1._get_holes(board_empty) == self.g2.get_holes(board_empty), f"g1: {self.g1._get_holes(board_empty)}, g2:{self.g2.get_holes(board_empty)}"

        for epsilon in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for i in range(1000):
                board = self.random_board(epsilon)
                assert (self.g1._get_holes(board) == self.g2.get_holes(board))

        print("PASSED: test_holes")

    def test_states(self):
        board_empty = [[0] * self.width for i in range(self.height)]
        assert (torch.all(self.g1.get_state(board_empty).eq(self.g2.get_state_properties(board_empty))))

        for epsilon in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for i in range(1000):
                board = self.random_board(epsilon, lines=i > 800)
                board_copy = deepcopy(board)
                board2 = deepcopy(board)
                board2_copy = deepcopy(board2)

                # state 1 not changes board
                state1 = self.g1.get_state(board)
                # for a in range(len(board)):
                #     for b in range(len(board[0])):
                #         assert board[a][b] == board_copy[a][b], f"a={a}, b={b}"
                #
                # state 2 does change board
                state2 = self.g2.get_state_properties(board2)
                # for a in range(len(board)):
                #     for b in range(len(board[0])):
                #         assert board2_copy[a][b] == board2[a][b], f"a={a}, b={b}"


                assert torch.all(state1.eq(state2)), f"eps: {epsilon}, i: {i}, g1: {state1}, g2: {state2}"

        print("PASSED: test_states")


def test():
    # if torch.cuda.is_available():
    #     model = torch.load("./model/model.pth")
    # else:
    #     model = torch.load("./model/model.pth", map_location=lambda storage, loc: storage)
    model = DeepQNetwork(4, 256, 1, 1)
    model.load_state_dict(torch.load("./model/model.pth"))
    model.eval()
    env = g1.TetrisGameAI()
    env.reset()
    while True:
        next_steps = env.get_all_possible_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.play_step(action)

        if done:
            break

if __name__ == '__main__':
    # test = Test()
    # random.seed(333)
    # test.test_holes()
    # test.test_states()

    test()
