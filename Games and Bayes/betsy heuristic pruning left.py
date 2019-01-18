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

split = list(input)
player = "x"

array = []

initial = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]
chunks = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]

def opponent(player):
    if player == "x": 
        return "o"
    else: 
        return "x"  

def printable_board(chunks):    
    board = ('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in chunks]))
    print(board)

"Recommendation"

def successor(initial,player):
    array[:] = []
    for dr in range(-x,x+1):
        if dr > 0:
            drop_command(abs(dr),player,initial,dr)
        elif dr < 0:
            rotate_command(abs(dr),player,initial,dr) 
        dr = dr+1
    return array

"Drop Command"

def drop_command(col_chosen,player,initial,dr):
    for i in range(0,n+3):
        if initial[i][col_chosen-1] != ".":
            array.append(copy.deepcopy(initial))
            array[-1][i-1][col_chosen-1] = player
            return array[dr+2]

"Citation: Aravind Parappil"

"Rotate Command"

def rotate_command(col_chosen,player,initial,dr):
   npboard=np.array(initial)
   if(len(np.where(npboard[:,col_chosen-1] == '.')[0]) > 0):
       spot = max(np.where(npboard[:,col_chosen-1] == '.')[0].tolist())+1
   else:
       spot = 0
   npboard[spot:, col_chosen-1] = np.roll(npboard[spot:,col_chosen-1], 1)
   array.append(npboard.tolist())
   npboard = np.array(initial)
   return array[dr+3]

def count function(chunks,player,n):
    
    
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
    








#------------------------------------------------------------------------------------------#
def evaluate(chunks,player,n):
    
    turn = player
    enemy = opponent(player)
    
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

def minimax(board, depth, isMax): 
    
    if (isMax == True):
        chance = 'x'
    else:
        chance = 'o'
    
    score = evaluate(board,chance,n)
    
    if(score == 10 or depth == 1):
        print 
        "goal or max depth reached"
        return score
        
    if (isMax == True):
        best = -1000
        for b in successor(board,chance):
            print "maxarray",b
            print "player",chance
            best = max(best, minimax(b, depth+1, False))
            print 'depth',depth
            print "first job done"
        return best

    else:
        best = 1000
        for b in successor(board,chance):
            print "minarray",b
            print "player",chance
            best = min(best, minimax(b, depth+1, True))
            print 'depth',depth
            print "second job done"
        return best


def solve(initial,player):
    minimax_values = []
    rec_array = ['Rotate column 3','Rotate column 2','Rotate column 1','Drop column 1','Drop column 2','Drop column 3']
    for b in successor(initial,player):
        print "successor",b
        minimax_values.append(minimax(b,0,True))
        maximum = max(minimax_values)
    for i in range(0,len(minimax_values)):
        if maximum == minimax_values[i]:
            return rec_array[i] 
        
print(solve(initial,player))