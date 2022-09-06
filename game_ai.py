import pygame
import random
import numpy as np
import torch
from copy import deepcopy

from settings import *

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# shapes
shape_list = [
    [0, 0, 0,
     1, 1, 1,
     0, 0, 0],  # rots = 1

    [0, 0, 0,
     0, 1, 0,
     1, 1, 1],  # rots = 3

    [0, 0, 0,
     0, 1, 0,
     0, 0, 0],  # rots = 0

    [0, 0, 0,
     1, 1, 0,
     0, 1, 1],  # rots = 1

    [0, 0, 0,
     0, 1, 1,
     1, 1, 0],  # rots = 1

    [0, 1, 0,
     0, 1, 0,
     1, 1, 0],  # rots = 3

    [0, 1, 0,
     0, 1, 0,
     0, 0, 0],  # rots = 1

    [0, 0, 0,
     0, 1, 1,
     0, 1, 1]  # rots = 0
]

color_list = [BLUE, YELLOW, GREEN, LIGHT_GREY, PINK, ORANGE]


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Block:
    def __init__(self, x, y, shape: list, shape_id: int, color_id: int):
        self.x = x
        self.y = y
        self.color_id = color_id
        self.shape_id = 0

        # translate shape
        self.shape = []
        for i in range(9):
            if shape[i] == 0:
                self.shape.append(None)
            else:
                # x
                if i % 3 == 0:
                    p_x = x - BLOCK_SIZE
                elif i % 3 == 1:
                    p_x = x
                else:
                    p_x = x + BLOCK_SIZE

                # y
                if i < 3:
                    p_y = y - BLOCK_SIZE
                elif i < 6:
                    p_y = y
                else:
                    p_y = y + BLOCK_SIZE

                self.shape.append(Point(p_x, p_y))

        self.i_x1, self.i_x2, self.i_y1, self.i_y2 = self._get_min_max_pos()

    def get_width_params(self) -> tuple:
        left, right = 0, 0
        for i in range(3):
            if self.shape[0 + i * 3] is not None:
                left = 1
                break

        for i in range(3):
            if self.shape[2 + i * 3] is not None:
                right = 1
                break

        return left, right

    def _get_min_max_pos(self):
        i_x1, i_x2, i_y1, i_y2 = None, None, None, None
        for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]:
            if self.shape[i] is not None:
                i_x1 = i
                break

        for i in [8, 5, 2, 7, 4, 1, 6, 3, 0]:
            if self.shape[i] is not None:
                i_x2 = i
                break

        for i in range(8):
            if self.shape[i] is not None:
                i_y1 = i
                break

        for i in range(8, -1, -1):
            if self.shape[i] is not None:
                i_y2 = i
                break

        return i_x1, i_x2, i_y1, i_y2

    def rotate(self):
        # reset all
        shape_copy = deepcopy(self.shape)
        self.shape = [None, None, None, None, None, None, None, None, None]

        if shape_copy[6] is not None:
            self.shape[0] = Point(shape_copy[6].x, shape_copy[6].y - 2 * BLOCK_SIZE)
        if shape_copy[3] is not None:
            self.shape[1] = Point(shape_copy[3].x + BLOCK_SIZE, shape_copy[3].y - BLOCK_SIZE)
        if shape_copy[0] is not None:
            self.shape[2] = Point(shape_copy[0].x + 2 * BLOCK_SIZE, shape_copy[0].y)
        if shape_copy[7] is not None:
            self.shape[3] = Point(shape_copy[7].x - BLOCK_SIZE, shape_copy[7].y - BLOCK_SIZE)
        if shape_copy[4] is not None:
            self.shape[4] = Point(shape_copy[4].x, shape_copy[4].y)
        if shape_copy[1] is not None:
            self.shape[5] = Point(shape_copy[1].x + BLOCK_SIZE, shape_copy[1].y + BLOCK_SIZE)
        if shape_copy[8] is not None:
            self.shape[6] = Point(shape_copy[8].x - 2 * BLOCK_SIZE, shape_copy[8].y)
        if shape_copy[5] is not None:
            self.shape[7] = Point(shape_copy[5].x - BLOCK_SIZE, shape_copy[5].y + BLOCK_SIZE)
        if shape_copy[2] is not None:
            self.shape[8] = Point(shape_copy[2].x, shape_copy[2].y + 2 * BLOCK_SIZE)

        self.i_x1, self.i_x2, self.i_y1, self.i_y2 = self._get_min_max_pos()

    def get_blocks_num(self):
        count = 0
        for p in self.shape:
            if p is not None:
                count += 1
        return count


def get_random_shape_color():
    shape_id = random.randint(0, len(shape_list) - 1)
    shape = shape_list[shape_id]
    color_id = random.randint(0, len(color_list) - 1)
    color = color_list[color_id]
    return shape, shape_id, color, color_id


def get_color_by_id(id: int):
    return color_list[id]


def color_symbol_to_color_id(color_symbol: int):
    return color_symbol - 1


def color_id_to_color_symbol(color_id: int):
    return color_id + 1


def get_rotations_num_by_shape_id(id: int):
    possible_rotations = 0
    if id in [0, 3, 4, 6]:
        possible_rotations = 1
    elif id in [1, 5]:
        possible_rotations = 3

    return possible_rotations


class TetrisGameAI:

    def __init__(self, w=WIDTH, h=HEIGHT, boundary=BOUNDARY, reset=True):
        self.w = w
        self.h = h
        self.height = self.h / BLOCK_SIZE
        self.boundary = boundary

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()
        if reset:
            self.reset()

    def reset(self):
        # init game state
        self.current_block = None
        self.blocks = []
        self.score = 0
        self.board = [[0] * int(self.w / BLOCK_SIZE) for i in range(int(self.h / BLOCK_SIZE))]
        self.cleared_lines = 0
        self.tetrominoes = 0
        self._summon_block()

    def _summon_block(self):
        x = self.w / 2
        y = BLOCK_SIZE
        shape, shape_id, color, color_id = get_random_shape_color()
        self.current_block = Block(x, y, shape, shape_id, color_id)

    def play_step(self, action):
        #  1. get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        #  2. move current block (until placed)
        self._move(action)  # move block L/R or rotate, and then place all the way down

        #  3. check for game over
        game_over = self._is_game_over()

        #  4. place block on board
        self._place_block_on_board(self.current_block, self.board)

        #  5. clear strikes
        idx_to_clear = self._check_cleared_lines(self.board)
        cleared_lines = len(idx_to_clear)
        self._clear_lines(self.board, idx_to_clear)
        score = 1 + (cleared_lines ** 2) * int(self.w / BLOCK_SIZE)
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += cleared_lines

        #  6. place new block
        if not game_over:
            self._summon_block()
        else:
            self.score -= 2

        #  7. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        #  8. return
        return score, game_over

    def _move(self, action: list):
        # [rotations, x]

        x = action[0]
        rotate_num = action[1]

        # rotate
        if rotate_num % 4 != 0:
            for i in range(rotate_num):
                self.current_block.rotate()

        # move to x
        diff = x - (self.current_block.shape[4].x / BLOCK_SIZE)
        for p in self.current_block.shape:
            if p is not None:
                p.x += diff * BLOCK_SIZE

        # release block until placed
        self._release_block()

    def _release_block(self) -> None:
        self._update_ui()
        self.clock.tick(SPEED)

        # increment y every frame and check: locked, game over
        while True:
            if self._is_block_locked(self.current_block, self.board):
                self.blocks.append(self.current_block)
                break

            # y++
            for p in self.current_block.shape:
                if p is not None:
                    p.y += BLOCK_SIZE

            if not DEBUGGING:
                self._update_ui()
                self.clock.tick(SPEED)

    def _is_game_over(self):
        for p in self.current_block.shape:
            if p is not None and p.y <= BOUNDARY:
                return True
        return False

    def _is_block_locked(self, block, board):
        y = block.shape[block.i_y2].y + BLOCK_SIZE
        if y >= self.h:
            return True

        for p in block.shape:
            if p is not None:
                y_p = int(p.y / BLOCK_SIZE) + 1
                x_p = int(p.x / BLOCK_SIZE)
                if 0 <= y_p < len(board) and board[y_p][x_p] != 0:
                    return True

        return False

    def get_all_possible_states(self) -> dict:
        states = dict()

        possible_rotations = 0
        if self.current_block.shape_id in [0, 3]:
            possible_rotations = 1
        elif self.current_block.shape_id in [1, 4]:
            possible_rotations = 3

        block_copy = deepcopy(self.current_block)
        for rot_num in range(possible_rotations + 1):
            left, right = block_copy.get_width_params()
            possible_x = [0 + left, int(self.w / BLOCK_SIZE) - 1 - right]
            for x in range(possible_x[0], possible_x[1] + 1):  # for each possible x, place and get state
                board_copy = deepcopy(self.board)

                # move to x, reset y
                mid_x = block_copy.shape[4].x
                for i, p in enumerate(block_copy.shape):
                    if p is not None:
                        p.x = x * BLOCK_SIZE - mid_x + p.x
                        if i < 3:
                            p.y = -2 * BLOCK_SIZE
                        elif i < 6:
                            p.y = -BLOCK_SIZE
                        else:
                            p.y = 0

                # place block down
                while not self._is_block_locked(block_copy, board_copy):
                    for p in block_copy.shape:
                        if p is not None:
                            p.y += BLOCK_SIZE

                # store on board copy
                self._place_block_on_board(block_copy, board_copy)

                # store states
                states[(x, rot_num)] = self.get_state(board_copy)

            # rotate
            block_copy.rotate()
        return states

    def _fill_lines_and_check_strike(self, block) -> int:
        # fill
        y_set = set()
        points = [p for p in block.shape if p is not None]
        for p in points:
            self.board[int(p.y / BLOCK_SIZE)][int(p.x / BLOCK_SIZE)] = 1
            y_set.add(p.y)

        # check for strike in changed y
        cleared = 0
        for y in y_set:
            line = self.board[int(y / BLOCK_SIZE)]
            strike = True
            for c in line:
                if c == 0:
                    strike = False
                    break
            if strike:
                for b in self.blocks:
                    for i in range(len(b.shape)):
                        if b.shape[i] is not None and b.shape[i].y == y:
                            b.shape[i] = None

                self.board[int(y / BLOCK_SIZE)] = [0] * int(self.w / BLOCK_SIZE)
                cleared += 1

        # clean empty blocks
        i = 0
        while i < len(self.blocks):
            if self.blocks[i].get_blocks_num() == 0:
                del self.blocks[i]
            else:
                i += 1

        return cleared

    def _place_block_on_board(self, block: Block, board: list):
        points = [p for p in block.shape if p is not None]
        color_symbol = color_id_to_color_symbol(block.color_id)
        for p in points:
            if int(p.y / BLOCK_SIZE) >= len(board):
                quit()
            board[int(p.y / BLOCK_SIZE)][int(p.x / BLOCK_SIZE)] = color_symbol

    def _update_ui(self):
        if DEBUGGING:
            return

        self.display.fill(BLACK)

        # draw boundary
        pygame.draw.line(self.display, RED, (0, BOUNDARY), (self.w, BOUNDARY), 1)

        for i, lines in enumerate(self.board):
            for j, cell in enumerate(self.board[i]):
                if cell != 0:
                    color = get_color_by_id(color_symbol_to_color_id(cell))
                    x, y = j * BLOCK_SIZE, i * BLOCK_SIZE
                    pygame.draw.rect(self.display, BORDER_COLOR, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, color, pygame.Rect(x, y, BLOCK_SIZE - 1, BLOCK_SIZE - 1))

        # draw current block
        color = get_color_by_id(self.current_block.color_id)
        for p in self.current_block.shape:
            if p is not None:
                pygame.draw.rect(self.display, BORDER_COLOR, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, color, pygame.Rect(p.x, p.y, BLOCK_SIZE - 1, BLOCK_SIZE - 1))

        # draw text
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _check_cleared_lines(self, board) -> list:
        lines_idx_to_delete = []
        for i, line in enumerate(board[::-1]):  # scan from bottom to top
            if 0 not in line:
                lines_idx_to_delete.append(len(board) - i - 1)
        return lines_idx_to_delete

    def _clear_lines(self, board, idx_list) -> None:
        for i in idx_list[::-1]:
            del board[i]
            board.insert(0, [0] * int(self.w / BLOCK_SIZE))

    def _get_holes(self, board) -> int:
        holes = 0
        for col in zip(*board):
            i = 0
            while i < len(col) and col[i] == 0:
                i += 1
            holes += len([cell for cell in col[i + 1:] if cell == 0])
        return holes

    def _get_bumpiness_total_heights(self, board):
        # bumpiness = sum of diff of heights of each column
        lines = np.array(board)
        mask = lines != 0
        height_per_col = self.height - np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        diff = np.abs(height_per_col[:-1] - height_per_col[1:])
        return np.sum(diff), np.sum(height_per_col)

    def get_state(self, board):
        lines_cleared_idx = self._check_cleared_lines(board)
        lines_cleared = len(lines_cleared_idx)
        self._clear_lines(board, lines_cleared_idx)
        holes = self._get_holes(board)
        bumpiness, total_heights = self._get_bumpiness_total_heights(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, total_heights])
