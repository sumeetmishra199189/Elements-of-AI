#---Elements of AI--

#Assignment0

#1.
# There are Abstraction used by the supplied code
#a) Set of valid subsets
#b)Inital State
#c)Successor function
#d)Cost function
#e)Goal state
#2 BFS Search

import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] )

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] )

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "R" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state
def successors(board):
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors( fringe.pop(0) ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[1])

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solve(initial_board)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")

#I am getting results for N=3 and N=4 very quickly, N=5 taking some time to produce result. But N=6 and higher values of N doesn't seem to produce result. This is because of time complexity of BFS, which is (b^d,b=branching factor,d=depth of tree). With higher branches and depth it takes much more time than usual.

#3Successor function2

def successors2(board):
 if (count_pieces(board) < N):
   for r in range(0, N):
     for c in range(0,N):
       #if (count_pieces(board) < N):
         if board[r][c]==0:
           #print([ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ])
           return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]
 return
