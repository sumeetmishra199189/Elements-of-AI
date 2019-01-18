#!/bin/python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018


#1. The below program takes a 16 board(scrambled board) as input and try to solve it in minimum steps

#2.The goal state is already known which is a sorted list of numbers from 1 to 16

#3.Solve Function: In the solve function algorithm 2 was applied with an additional condition to discard already visited states. A python dictionary closed was declared and tuple(state) and priority of that board was inserted as key value pair. If the successor board already visited, it will not be explored again(in every move L,R,U,D, a previous board was reached. Idea was to discard that board). If the successor was in fringe and not explored ,it will still be put in fringe and removed from fringe based on their priority(fringe was taken as priority queue).

#4. Heuristic function: No. of misplaced board and manhattan distance code is given below, but was not used in final solution because both of them are inadmissible for this board. If one tile was moved down then the misplaced tiles gave 4 and manhattan distance gave even more greater number whereas it could be solved in 1 step. No of misplaced tiles across diagonals also not used as it takes too much time.
#used heuristic: 2 heuristic function was used which are Manhattan distance divided by 4(optimal but time consuming) and calculating board cost=(row cost+column cost) divided by 2. Both of them are admissible. Because 1 move can displace 4 tiles so manhatten distance was divided by 4 is admissible. Board cost calculates row cost and column cost as min(max(left move),max(right move)) across a row for each row and min(max(up move),max(down move)) across a column for each column. The sum was divided by 2 for any overlap. In this heuristic function(calculate_heuristic_1) which calls 2 more functions row_cost and col_cost which calculates the costs for row and column. Conditions were used in both to handle some issues that observed during debugging the logic. Logic explanation given near code. Final heuristic retained was board cost.

#5.Interesting findings:The 12 board problem was solved in seconds but takes 16 moves ,if the divided by 2 was removed from board cost heuristic. Otherwise it seems to hang finding for an optimal solution. The divided by 2 logic was used as it makes the heuristic admissible and gives optimal solution at expence of some time.


#Citations-
#1.Priority Queue-https://dbader.org/blog/priority-queues-in-python
#2.Python dictionary-https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/
#3.Board Cost logic-from Piazza Discussion under name (Let's start heuristic function discussion)






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
        #a=get_distance(pretty_board[i][i],i,i)
        # print(a)
        heuristic_value_1 += get_distance_1(pretty_board[i][i],i,i)
        heuristic_value_2 += get_distance_1(pretty_board[j][i],j,i)
        j=j-1
    heuristic_value=max(heuristic_value_1,heuristic_value_2)
    return (heuristic_value+cost)
#get distance1 function gets inputs as a value and then we calculate it's manhattan distance
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

#manhattan distance calculates the manhattan distance of the entire board that it received as input
def Manhattan_distance(value):
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
#row cost calculates the manhattan distance in row by moving 1 element to left and 1 element to right and taking the minimum of that
def row_cost(value):
    goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    board=board_converter(value)
    l_move=[]
    r_move=[]
    for i in range(0,4):
         for j in range(0,4):
            if  (board[i][j])==goal_state[i][j] :     #checks if tile is in correct place then put cost as 0
                x=0
                l_move.append(x)
                r_move.append(x)
            elif((board[i][j])%4==0 and abs(j-(((board[i][j])%4)-1))>3):  #checks for the edges like(4,8,12,16) and handles if the value is more than 3
                x=0
                l_move.append(x)
                r_move.append(x)
            else:
                x=abs(j-(((board[i][j])%4)-1))
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
#column cost calculates the manhattan distance in column by moving 1 element up and 1 element down and taking the minimum of that
def col_cost(value):
    goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    board=board_converter(value)
    u_move=[]
    d_move=[]
    res_move1=[1,2,3]
    res_move2=[5,6,7]
    for i in range(0,4):
         for j in range(0,4):
             if (board[j][i])==goal_state[j][i] :   #checks if tile is in correct place then put cost as 0
                x=0
                u_move.append(x)
                d_move.append(x)
             elif(board[j][i]%4==0):               #checks for (4,8,12,16) and handles the cost by deducting 1
                x=abs(abs(j-(int((board[j][i])/4)))-1)
                d_move.append(x)
                u_move.append(abs(4-x))
             elif(j==2 and board[j][i] in res_move2):  #checks if 5,6,7 in 3rd row then swap the up move and down move
                x=abs(j-(int((board[j][i])/4)))
                u_move.append(x)
                d_move.append(abs(4-x))
             elif(j==3 and board[j][i] in res_move1):   #checks if 1,2,3 in 4rd row then swap the up move and down move
                x=abs(j-(int((board[j][i])/4)))
                u_move.append(x)
                d_move.append(abs(4-x))
             else:
                x=abs(j-(int((board[j][i])/4)))
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
def calculate_heuristic_1(board,cost_so_far):
    heuristic_value=0
    heuristic_value=math.ceil((row_cost(board)+col_cost(board))/2)
    #heuristic_value=Manhattan_distance(board)
    cost=cost_so_far
    return (heuristic_value+cost)





# The solver! - using A*
def solve(initial_board):
    fringe = PriorityQueue()
    closed = {}
    priority_1=calculate_heuristic_1(initial_board,0)
    fringe.put((priority_1,(initial_board,"")))
    while not fringe.empty():
        (priority,(state, route_so_far))=fringe.get()
        closed[tuple(state)] = priority
        if is_goal(state):
          return(route_so_far)
        for (succ, move) in successors( state ):
            route=str( route_so_far + " " + move )
            cost_so_far=len(route_so_far.split())
            priority_s=calculate_heuristic_1(succ,cost_so_far)
            if tuple(succ) not in closed:
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
