import pygame
import random
from copy import deepcopy
from settings import *

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Block:
    def __init__(self, x, y, shape, color):
        self.x = x
        self.y = y
        self.color = color

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

        self.mid = self.shape[4]
        self.i_x1, self.i_x2, self.i_y1, self.i_y2 = self._get_min_max_pos()
        self.prev_rotation = None
        self.prev_x1, self.prev_x2, self.prev_y1, self.prev_y2 = None, None, None, None

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
        self.prev_rotation = self.shape
        self.prev_x1, self.prev_x2, self.prev_y1, self.prev_y2 = self.i_x1, self.i_x2, self.i_y1, self.i_y2

        # reset all
        self.shape = [None, None, None, None, None, None, None, None, None]

        if self.prev_rotation[6] is not None:
            self.shape[0] = Point(self.prev_rotation[6].x, self.prev_rotation[6].y - 2*BLOCK_SIZE)
        if self.prev_rotation[3] is not None:
            self.shape[1] = Point(self.prev_rotation[3].x + BLOCK_SIZE, self.prev_rotation[3].y - BLOCK_SIZE)
        if self.prev_rotation[0] is not None:
            self.shape[2] = Point(self.prev_rotation[0].x + 2 * BLOCK_SIZE, self.prev_rotation[0].y)
        if self.prev_rotation[7] is not None:
            self.shape[3] = Point(self.prev_rotation[7].x - BLOCK_SIZE, self.prev_rotation[7].y - BLOCK_SIZE)
        if self.prev_rotation[4] is not None:
            self.shape[4] = Point(self.prev_rotation[4].x, self.prev_rotation[4].y)
        if self.prev_rotation[1] is not None:
            self.shape[5] = Point(self.prev_rotation[1].x + BLOCK_SIZE, self.prev_rotation[1].y + BLOCK_SIZE)
        if self.prev_rotation[8] is not None:
            self.shape[6] = Point(self.prev_rotation[8].x - 2 * BLOCK_SIZE, self.prev_rotation[8].y)
        if self.prev_rotation[5] is not None:
            self.shape[7] = Point(self.prev_rotation[5].x - BLOCK_SIZE, self.prev_rotation[5].y + BLOCK_SIZE)
        if self.prev_rotation[2] is not None:
            self.shape[8] = Point(self.prev_rotation[2].x, self.prev_rotation[2].y + 2 * BLOCK_SIZE)

        self.i_x1, self.i_x2, self.i_y1, self.i_y2 = self._get_min_max_pos()

    def revert_rotation(self):
        self.shape = self.prev_rotation
        self.i_x1, self.i_x2, self.i_y1, self.i_y2 = self.prev_x1, self.prev_x2, self.prev_y1, self.prev_y2

    def get_blocks_num(self):
        count = 0
        for p in self.shape:
            if p is not None:
                count += 1
        return count


def get_random_shape_color():
    r = random.randint(0, len(SHAPE_LST) - 1)
    shape = SHAPE_LST[r]
    r = random.randint(0, len(COLOR_LST) - 1)
    color = COLOR_LST[r]
    return shape, color


class TetrisGameAI:

    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()
        # self.reset()

        # init game state
        self.current_block = None
        self.blocks = []
        self.score = 0
        self.direction = 0
        self.lines = [[0] * int(self.w / BLOCK_SIZE) for i in range(int(self.h / BLOCK_SIZE))]

        self._summon_block()

    def _summon_block(self):
        x = self.w / 2
        y = -BLOCK_SIZE
        shape, color = get_random_shape_color()
        self.current_block = Block(x, y, shape, color)

    def play_step(self):
        #  0. reset direction and rotation
        self.direction = 0

        #  1. get user input & rotate
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.direction = 2
                elif event.key == pygame.K_d:
                    self.direction = 1
                elif event.key == pygame.K_a:
                    self.direction = -1

        #  2. move current block
        self._move(self.direction)  # move block L/R or rotate

        #  3+4. place new block if locked, and check for game over
        if self._is_block_locked():
            self.direction = 0
            self.blocks.append(self.current_block)

            # check for game over
            if self._is_collision_top():
                return True, self.score

            self._fill_lines_and_check_strike(self.current_block)
            self._summon_block()

        #  5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        #  6. return
        return False, self.score

    def _move(self, direction):
        # move y+1
        for p in self.current_block.shape:
            if p is not None:
                p.y += BLOCK_SIZE

        # try to rotate
        if direction == Direction.ROTATE:
            self.current_block.rotate()

            # 1. check if collides after rotation
            can_rotate = True
            for p in self.current_block.shape:
                if p is not None:
                    y_p = int(p.y / BLOCK_SIZE)
                    x_p = int(p.x / BLOCK_SIZE)
                    if self.lines[y_p][x_p] == 1 or p.x <= 0 - BLOCK_SIZE or p.x >= self.w:  # if occupied/out of boarder
                        can_rotate = False
                        break

            # 2. if cannot rotate, revert
            if not can_rotate:
                self.current_block.revert_rotation()

        # try to move left/right
        elif direction == Direction.RIGHT or direction == Direction.LEFT:
            # 1. check if can move
            can_move = True
            for p in self.current_block.shape:
                if p is not None and not 0 <= p.x + direction * BLOCK_SIZE <= self.w - BLOCK_SIZE:
                    can_move = False
                    break

            # 2. if can, move every block in shape as a group
            if can_move:
                for p in self.current_block.shape:
                    if p is not None:
                        p.x += direction * BLOCK_SIZE

        self.current_block.x = self.current_block.mid.x
        self.current_block.y = self.current_block.mid.y

    def _is_collision_top(self):
        for p in self.current_block.shape:
            if p is not None and p.y <= BOUNDARY:
                return True
        return False

    def _is_block_locked(self):
        y = self.current_block.shape[self.current_block.i_y2].y + BLOCK_SIZE
        if y >= self.h:
            return True

        for p in self.current_block.shape:
            if p is not None:
                y_p = int(p.y / BLOCK_SIZE) + 1
                x_p = int(p.x / BLOCK_SIZE)
                if self.lines[y_p][x_p] == 1:
                    return True
        return False

    def _fill_lines_and_check_strike(self, block):
        # fill
        y_set = set()
        points = [p for p in block.shape if p is not None]
        for p in points:
            self.lines[int(p.y / BLOCK_SIZE)][int(p.x / BLOCK_SIZE)] = 1
            y_set.add(p.y)

        # check for strike in changed y
        for y in y_set:
            line = self.lines[int(y / BLOCK_SIZE)]
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

                self.lines[int(y / BLOCK_SIZE)] = [0] * int(self.w / BLOCK_SIZE)
                self.score += 1

        # clean empty blocks
        i = 0
        while i < len(self.blocks):
            if self.blocks[i].get_blocks_num() == 0:
                del self.blocks[i]
            else:
                i += 1

    def _update_ui(self):
        self.display.fill(BLACK)

        # draw boundary
        pygame.draw.line(self.display, RED, (0, BOUNDARY), (self.w, BOUNDARY), 1)

        # draw locked blocks
        for b in self.blocks:
            for p in b.shape:
                if p is not None:
                    pygame.draw.rect(self.display, BORDER_COLOR, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, b.color, pygame.Rect(p.x, p.y, BLOCK_SIZE - 1, BLOCK_SIZE - 1))

        # draw current block
        for p in self.current_block.shape:
            if p is not None:
                pygame.draw.rect(self.display, BORDER_COLOR, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, self.current_block.color, pygame.Rect(p.x, p.y, BLOCK_SIZE - 1, BLOCK_SIZE - 1))

        # draw text
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # def reset(self):
    #     # init game state
    #     self.frame_iteration = 0


if __name__ == '__main__':
    game = TetrisGameAI()

    # game loop
    while True:
        game_over, score = game.play_step()
        if game_over:
            break

    print('Final Score', score)

    pygame.quit()
