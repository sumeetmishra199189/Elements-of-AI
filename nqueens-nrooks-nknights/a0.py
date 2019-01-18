#!/usr/bin/env python
import sys


# Count # of pieces in given row
def count_on_row(board, row):
    return sum(board[row])


# Count # of pieces in given column
def count_on_col(board, col):
    return sum([row[col] for row in board])


# Count total # of pieces on board
def count_pieces(board):
    return sum([sum(row) for row in board])


# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    if (r_q_k == 'nrook'):
        str = ""
        for row in range(0, N):
            for square in range(0, N):
                if not restriction(row, square):
                    str += "X "
                elif board[row][square] == 1:
                    str += "R "
                else:
                    str += "_ "
            str += "\n"
        return str
        # return "\n".join([ " ".join([ "R" if col else "_" for col in row ]) for row in board])
    elif (r_q_k == 'nqueen'):
        str = ""
        for row in range(0, N):
            for square in range(0, N):
                if not restriction(row, square):
                    str += "X "
                elif board[row][square] == 1:
                    str += "Q "
                else:
                    str += "_ "
            str += "\n"
        return str
    # return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])
    elif (r_q_k == 'nknight'):
        str = ""
        for row in range(0, N):
            for square in range(0, N):
                if not restriction(row, square):
                    str += "X "
                elif board[row][square] == 1:
                    str += "K "
                else:
                    str += "_ "
            str += "\n"
        return str
    # return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])


# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1, ] + board[row][col + 1:]] + board[row + 1:]


# Get list of successors of given board state
def successor_r(board):
    c_pieces = count_pieces(board)
    sucs = []
    for r in range(0, N):
        if (count_pieces(board) == 0 and restriction(r, 0)):
            sucs.append(add_piece(board, r, 0))
        elif (count_pieces(board) < N and sum(board[r]) == 0 and restriction(r, count_pieces(board))):
            # for r in range(0, N):
            sucs.append(add_piece(board, r, count_pieces(board)))
            # return sucs
    return sucs


def successor_q(board):
    c_pieces = count_pieces(board)
    sucs = []
    for r in range(0, N):
        if (c_pieces == 0 and restriction(r, 0)):
            return [add_piece(board, r, 0) for r in range(0, N)]
        elif (count_pieces(board) < N and (sum(board[r]) == 0) and check_left_upper_diag(board, r, c_pieces,
                                                                                         N) and check_right_upper_diag(
            board, r, c_pieces, N) and check_left_lower_diag(board, r, c_pieces, N) and check_right_lower_diag(
            board, r, c_pieces, N) and restriction(r, c_pieces)):
            sucs.append(add_piece(board, r, count_pieces(board)))
    return sucs
def valid_moves_k(board,r,c):
    r_moves = [2, 1, -1, -2, -2, -1, 1, 2]  # from geek for geek website
    c_moves = [1, 2, 2, 1, -1, -2, -2, -1]  # from geek for geek website
    for i in range(0, 8):
        r1 = r + r_moves[i]
        c1 = c + c_moves[i]
        if (r1 >= 0 and c1 >= 0 and r1 < N and c1 < N):
            if board[r1][c1] == 1:
                return False
    return True
def successor_k(board):
    sucs = []
    if (count_pieces(board) < N):
        for r in range(0, N):
            for c in range(0, N):
                if (restriction(r, c) and (board[r][c] == 0) and valid_moves_k(board,r,c)):
                    sucs.append(add_piece(board, r, c))

    return sucs

#The diagonal logic for queen is refered from geek for geek website where the code was in C
def check_left_upper_diag(board, r, c, N):
    while (r >= 0 and c >= 0):
        if (board[r][c]) == 1:
            return False
        r = r - 1
        c = c - 1
    return True


def check_right_upper_diag(board, r, c, N):
    while (r >= 0 and c < N):
        if (board[r][c]) == 1:
            return False
        r = r - 1
        c = c + 1
    return True


def check_left_lower_diag(board, r, c, N):
    while (r < N and c >= 0):
        if (board[r][c]) == 1:
            return False
        r = r + 1
        c = c - 1
    return True


def check_right_lower_diag(board, r, c, N):
    while (r < N and c < N):
        if (board[r][c]) == 1:
            return False
        r = r + 1
        c = c + 1
    return True


# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and all([count_on_row(board, r) <= 1 for r in range(0, N)]) and all(
        [count_on_col(board, c) <= 1 for c in range(0, N)])


# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        if r_q_k == 'nrook':
            for s in successor_r(fringe.pop()):
                if is_goal(s):
                    return (s)
                fringe.append(s)
        elif r_q_k == 'nqueen':
            for s in successor_q(fringe.pop()):
                if is_goal(s):
                    return (s)
                fringe.append(s)
        elif r_q_k == 'nknight':
            for s in successor_k(fringe.pop()):
                if is_goal(s):
                    return (s)
                fringe.append(s)
    return False


# This is N, the size of the board. It is passed through command line arguments.
r_q_k = (sys.argv[1])
N = int(sys.argv[2])
count = int(sys.argv[3])


def restriction(r, c):
    res = []
    r_res = []
    c_res = []
    res_final = []
    r1 = r
    c1 = c
    s_p = 4
    # i=0
    for i in range(0, count * 2):
        res.append(sys.argv[s_p + i])
    for i in range(0, len(res), 2):
        r_res.append(res[i])
    for i in range(1, len(res), 2):
        c_res.append(res[i])

    for j in range(0, len(r_res)):
        res_final.append([int(r_res[j]) - 1, int(c_res[j]) - 1])
    for h in range(0, len(res_final)):
        if [r, c] == res_final[h]:
            return False
    return True


# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0] * N] * N
print("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solve(initial_board)
print(printable_board(solution) if solution else "Sorry, no solution found. :(")
