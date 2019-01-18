#!/bin/python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
#
from Queue import PriorityQueue
from random import randrange, sample
import sys
import string
import math
# shift a specified row left (-1) or right (+1)
def shift_row(state, row, dir):
    change_row = state[(row*4):(row*4+4)]
    return ( state[:(row*4)] + change_row[-dir:] + change_row[:-dir] + state[(row*4+4):], ("L" if dir == -1 else "R") + str(row+1) )

# shift a specified col up (-1) or down (+1)
def shift_col(state, col, dir):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]
    return (tuple(s), ("U" if dir == -1 else "D") + str(col+1) )

# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print '%3d %3d %3d %3d' % (row[j:(j+4)])

# return a list of possible successor states
def successors(state):
    return [ shift_row(state, i, d) for i in range(0,4) for d in (1,-1) ] + [ shift_col(state, i, d) for i in range(0,4) for d in (1,-1) ]

# just reverse the direction of a move name, i.e. U3 -> D3
def reverse_move(state):
    return state.translate(string.maketrans("UDLR", "DURL"))

# check if we've reached the goal

def is_goal(state):
    return sorted(state) == list(state)

#board converter converts my board into 2-D Board
def board_converter(state):
 board=[]
 for i in range(0,4):
     board.append(state[(i*4):(i*4+4)])
 return board    

# The heuristic function calculates manhatten distance and only checks for diagonal elements
def calculate_heuristic(board,cost):
    j=3
    heuristic_value = 0
    heuristic_value_1 = 0
    heuristic_value_2 = 0
    for i in range(0,4):
        pretty_board=board_converter(board)
        heuristic_value_1 += get_distance_1(pretty_board[i][i],i,i)
        heuristic_value_2 += get_distance_1(pretty_board[j][i],j,i)
    heuristic_value=max(heuristic_value_1,heuristic_value_2)
    return (heuristic_value+cost)
#get distance1 function gets inputs as a value and then we calculate it's manhatten distance
def get_distance_1(value,i,j):
    #goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    #print(goal_state)
    count=0
    x=abs(math.ceil((value)/4))
    y=abs(((value)%4)-1)
    count+=x+y
    return (count)

#get distance function counts the number of misplaced tiles
def get_distance(value, i, j):
    goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    #print(goal_state)
    count=0
    if(goal_state[i][j])!=value:
        count+=1
    
    return (count)
#manhatten distance calculates the manhatten distance of the entire board that it received as input
def Manhatten_distance(value):
    board=board_converter(value)    
    #goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    count=0
    for i in range(0,4):
        for j in range(0,4):
            x=abs(math.ceil((board[i][j])/4))
            y=abs(((board[i][j])%4)-1)
            count+=x+y
    count1=math.ceil(count/4)
    return count1       
#row cost calculates the manhatten distance in row by moving 1 element to left and 1 element to right and taking the minimum of that
def row_cost(value):
    #goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    board=board_converter(value)
    l_move=[]
    r_move=[]
    for i in range(0,4):
         for j in range(0,4):
            x=abs(((board[i][j])%4)-1)
            l_move.append(x)
            r_move.append(abs(4-x))
    o_cost_l=l_move[0:4]
    t_cost_l=l_move[4:8]
    th_cost_l=l_move[8:12]
    f_cost_l=l_move[12:16]

    o_cost_r=r_move[0:4]
    t_cost_r=r_move[4:8]
    th_cost_r=r_move[8:12]
    f_cost_r=r_move[12:16]
    one=min(max(o_cost_l),max(o_cost_r))
    two=min(max(t_cost_l),max(t_cost_r))
    three=min(max(th_cost_l),max(th_cost_r))
    four=min(max(f_cost_l),max(f_cost_r))

    r_cost=one+two+three+four
    return r_cost
#column cost calculates the manhatten distance in column by moving 1 element up and 1 element down and taking the minimum of that
def col_cost(value):
    #goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    board=board_converter(value)
    u_move=[]
    d_move=[]
    for i in range(0,4):
         for j in range(0,4):
            x=abs(int((board[i][j])/4))
            d_move.append(x)
            u_move.append(abs(4-x))
    o_cost_u=u_move[0:4]
    t_cost_u=u_move[4:8]
    th_cost_u=u_move[8:12]
    f_cost_u=u_move[12:16]

    o_cost_d=d_move[0:4]
    t_cost_d=d_move[4:8]
    th_cost_d=d_move[8:12]
    f_cost_d=d_move[12:16]
    one=min(max(o_cost_u),max(o_cost_d))
    two=min(max(t_cost_u),max(t_cost_d))
    three=min(max(th_cost_u),max(th_cost_d))
    four=min(max(f_cost_u),max(f_cost_d))
    c_cost=one+two+three+four 
    return c_cost

#calculate_heuristic1 functions receives row cost and column cost and add cost of the route takes so far to it and returns the cumulative sum
def calculate_heuristic_1(board,route):
    heuristic_value=0
    heuristic_value=math.ceil((row_cost(board)+col_cost(board))/4)
    #heuristic_value=Manhatten_distance(board)
    cost=route
    return (heuristic_value+cost)





# The solver! - using A*(implemented search algorithm 3 to discard visited states)

def solve(initial_board):
    fringe = PriorityQueue()
    closed = {}
    priority_1=calculate_heuristic_1(initial_board,0)
    fringe.put((priority_1,(initial_board,"")))
    while not fringe.empty():
        (priority,(state, route_so_far))=fringe.get()
        closed[tuple(state)] = priority
        #closed.append((priority,(state)))
        if is_goal(state):
            return(route_so_far)
        for (succ, move) in successors( state ):
            route=str( route_so_far + " " + move )
            cost_so_far=len(route_so_far.split())
            priority_s=calculate_heuristic_1(succ,cost_so_far)
            if tuple(succ) not in closed:
                fringe.put((priority_s,(succ,route)))
            elif tuple(succ) in closed and closed[tuple(succ)]<closed[tuple(state)]:
                fringe.put((priority_s,(succ,route)))
return False


#########################----------------######################


# test cases
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [ int(i) for i in line.split() ]

if len(start_state) != 16:
    print "Error: couldn't parse start state file"

print "Start state: "
print_board(tuple(start_state))

print "Solving..."
route = solve(tuple(start_state))
#print(route)
print "Solution found in " + str(len(route)/3) + " moves:" + "\n" + route
