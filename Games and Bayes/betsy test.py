#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:01:25 2018
@author: varunmiranda
Citations:
https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
https://stackoverflow.com/questions/17870612/printing-a-two-dimensional-array-in-python
https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-3-tic-tac-toe-ai-finding-optimal-move/
"""

import copy
import numpy as np

n = 3
x = n
input = "...x..o.ox.oxxxooo"
#input = "xoxoxoxoxoxoxoxoxo"

split = list(input)
turn = "x"

array = []

initial = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]
chunks = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]

def opponent():
    if turn == "x": 
        return "o"
    else: 
        return "x"  

enemy = opponent()

def printable_board(chunks):    
    board = ('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in chunks]))
    print(board)

"Recommendation"

def successor(initial):
    array[:] = []
    for dr in range(-x,x+1):
        if dr > 0:
            value = drop_command(abs(dr),initial,dr)
            goal_state(value,n,dr)
        elif dr < 0:
            value = rotate_command(abs(dr),initial,dr) 
            goal_state(value,n,dr)
        dr = dr+1
    return array

"Drop Command"

def drop_command(col_chosen,initial,dr):
    for i in range(0,n+3):
        if initial[i][col_chosen-1] != ".":
            array.append(copy.deepcopy(initial))
            array[-1][i-1][col_chosen-1] = turn
            return array[dr+2]

"Citation: Aravind Parappil"

"Rotate Command"

def rotate_command(col_chosen,initial,dr):
   npboard=np.array(initial)
   if(len(np.where(npboard[:,col_chosen-1] == '.')[0]) > 0):
       spot = max(np.where(npboard[:,col_chosen-1] == '.')[0].tolist())+1
   else:
       spot = 0
   npboard[spot:, col_chosen-1] = np.roll(npboard[spot:,col_chosen-1], 1)
   array.append(npboard.tolist())
   npboard = np.array(initial)
   return array[dr+3]

#------------------------------------------------------------------------------------------#

def goal_state(chunks,n,dr):
    if evaluate(chunks,n) == 10: 
        if dr < 0:
            print ('I would recommend rotating column '+str(abs(dr))+' and you will win')
        elif dr > 0:
            print ('I would recommend dropping a piece in column '+str(dr)+' and you will win')
        return True 

def evaluate(chunks,n):
    
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    
    for i1 in range(0,n):
        count_player=0
        count_opponent=0
        for j1 in range(0,n):
            if chunks[i1][j1]==turn:
                count_player+=1
            elif chunks[i1][j1]==enemy:
                count_opponent+=1
        if(count_player==n):
            score1 = 10
        elif(count_opponent==n):
            if i1<n-1:
                continue
            score1 = -10


    for j2 in range(0,n):
        count_player=0
        count_opponent=0
        for i2 in range(0,n):
            if chunks[i2][j2]==turn:
                count_player+=1
            elif chunks[i2][j2]==enemy:
                count_opponent+=1
        if(count_player==n):
            score2 = 10
        elif(count_opponent==n):
            if i2<n-1:
                continue
            score2 = -10
                
    i3=0
    j3=0
    count_player=0
    count_opponent=0
    while(i3<n):
        if chunks[i3][j3]==turn:
            count_player+=1
        elif chunks[i3][j3]==enemy:
            count_opponent+=1
        i3+=1
        j3+=1
        if(count_player==n):
            score3 = 10
        elif(count_opponent==n):
            if i3<n-1:
                continue
            score3 = -10

    i4=n-1
    j4=0
    count_player=0
    count_opponent=0
    while(i4>=0):
        if chunks[i4][j4]==turn:
            count_player+=1
        elif chunks[i4][j4]==enemy:
            count_opponent+=1
        i4-=1
        j4+=1
        if(count_player==n):
            score4 = 10
        elif(count_opponent==n):
            if i>0:
                continue
            score4 = -10
    
    if score1 == 10 or score2 == 10 or score3 == 10 or score4 == 10:
        return 10
    elif score1 == -10 or score2 == -10 or score3 == -10 or score4 == -10:
        return -10
    else:
        return 0

#------------------------------------------------------------------------------------------#

def minimax(board, depth, isMax,alpha,beta): 
    while depth <= 2:
        
            score = evaluate(board,n)
            print(score)
            
            if score == 10:
                return score
          
            if (isMax == True):
                best = -1000
                for b in successor(board):
                    print("maxarray",b)
                    best = max(best, minimax(b, depth+1, False)) 
                    alpha = max( alpha, best)
                    if beta <= alpha:
                        break
                    print("first job done")
                return best
        
            else:
                best = 1000
                for b in successor(board):
                    print("minarray",b)
                    best = min(best, minimax(b, depth+1, True))
                    beta = min( beta, best)
                    if beta <= alpha:
                        break
                    print("second job done")
                return best
    
'''
function minimax(board, depth, isMaximizingPlayer):

    if current board state is a terminal state :
        return value of the board
    
    if isMaximizingPlayer :
        bestVal = -INFINITY 
        for each move in board :
            value = minimax(board, depth+1, false)
            bestVal = max( bestVal, value) 
        return bestVal

    else :
        bestVal = +INFINITY 
        for each move in board :
            value = minimax(board, depth+1, true)
            bestVal = min( bestVal, value) 
        return bestVal
'''
minimax(initial,0,True)


#def solve(initial_board):
#    fringe = [initial_board]
#    while len(fringe) > 0:
#        for s in successor(fringe.pop()):
#            if goal_state(s,n) == True:
#                return(s)
#            fringe.append(s)
#    return False