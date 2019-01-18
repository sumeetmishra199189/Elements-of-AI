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
#input = ".................."

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
    
    if initial[n+2][col_chosen-1] == ".":
        array.append(copy.deepcopy(initial))
        array[-1][n+2][col_chosen-1] = player
        return array[dr+2]
    else:    
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

#------------------------------------------------------------------------------------------#


def evaluate(chunks,player,n):
    
    #Raw Heuristic
    
    turn = player
    enemy = opponent(player)
    score = 0
    
    count_player=0
    count_opponent=0
    for i in range(0,n+3):
        for j in range (0,n):
            if chunks[i][j] == turn:
                count_player+=1
            elif chunks[i][j] == enemy:
                count_opponent+=1
    score = count_player - count_opponent
                
    #Drop Logic - Almost goal - row
    
    for i in range(0,n):
        count_player = 0
        count_opponent = 0
        count_dots = 0
        for j in range(0,n):
            if chunks[i][j]==turn:
                count_player+=1
            elif chunks[i][j]==enemy:
                count_opponent+=1
            elif chunks[i][j]== '.' and chunks[i+1][j] != '.':
                count_dots+=1
        if(count_player==n-1 and count_dots==1):
            score = 50
        elif(count_opponent==n-1 and count_dots==1):
            score = -50
    
     #Drop Logic - Almost goal - column
    
    for j in range(0,n):
        count_player = 0
        count_opponent = 0
        count_dots = 0
        for i in range(0,n):
            if chunks[i][j]==turn:
                count_player+=1
            elif chunks[i][j]==enemy:
                count_opponent+=1
            elif chunks[i][j]== '.':
                count_dots+=1
        if(count_player==n-1 and count_dots==1):
            score = 50
        elif(count_opponent==n-1 and count_dots==1):
            if j<n-1:
                continue
            score = -50
    
    #Drop Logic - Almost goal - left diagonal
    
    i=0
    j=0
    count_player=0
    count_opponent=0
    count_dots = 0
    while(i<n):
        if chunks[i][j]==turn:
            count_player+=1
        elif chunks[i][j]==enemy:
            count_opponent+=1
        elif chunks[i][j]=='.' and chunks[i+1][j]!='.':
            count_dots+=1
        i+=1
        j+=1
        if(count_player==n-1 and count_dots == 1):
            score = 50
        elif(count_opponent==n-1 and count_dots == 1):
            score = -50
    
    #Drop Logic - Almost goal - right diagonal
    
    i=n-1
    j=0
    count_player=0
    count_opponent=0
    count_dots = 0
    while(i>=0):
        if chunks[i][j]==turn:
            count_player+=1
        elif chunks[i][j]==enemy:
            count_opponent+=1
        elif chunks[i][j]=='.' and chunks[i+1][j]!='.':
            count_dots+=1
        i-=1
        j+=1
        if(count_player==n-1 and count_dots == 1):
            score = 50
        elif(count_opponent==n-1 and count_dots == 1):
            score = -50
    
    #Goal checks
    
    for i in range(0,n):
        count_player=0
        count_opponent=0
        for j in range(0,n):
            if chunks[i][j]==turn:
                count_player+=1
            elif chunks[i][j]==enemy:
                count_opponent+=1
        if(count_player==n):
            score = 100
        elif(count_opponent==n):
            if i<n-1:
                continue
            score = -100


    for j in range(0,n):
        count_player=0
        count_opponent=0
        for i in range(0,n):
            if chunks[i][j]==turn:
                count_player+=1
            elif chunks[i][j]==enemy:
                count_opponent+=1
        if(count_player==n):
            score = 100
        elif(count_opponent==n):
            if j<n-1:
                continue
            score = -100
                
    i=0
    j=0
    count_player=0
    count_opponent=0
    while(i<n):
        if chunks[i][j]==turn:
            count_player+=1
        elif chunks[i][j]==enemy:
            count_opponent+=1
        i+=1
        j+=1
        if(count_player==n):
            score = 100
        elif(count_opponent==n):
            score = -100

    i=n-1
    j=0
    count_player=0
    count_opponent=0
    while(i>=0):
        if chunks[i][j]==turn:
            count_player+=1
        elif chunks[i][j]==enemy:
            count_opponent+=1
        i-=1
        j+=1
        if(count_player==n):
            score = 100
        elif(count_opponent==n):
            score = -100
    
    return score

#------------------------------------------------------------------------------------------#

def minimax(board, depth, isMax): 
    
    if (isMax == True):
        chance = player
    else:
        chance = opponent(player)
    
    score = evaluate(board,chance,n)
    
    if(score == 10 or depth == 1):
        return score
        
    if (isMax == True):
        best = -1000
        for b in successor(board,chance):
            best = max(best, minimax(b, depth+1, False))
        return best

    else:
        best = 1000
        for b in successor(board,chance):
            best = min(best, minimax(b, depth+1, True))
        return best


def solve(initial,player):
    minimax_values = []
    b_array = []
    rec_array = ['Rotate column 3','Rotate column 2','Rotate column 1','Drop column 1','Drop column 2','Drop column 3']
    
    for b in successor(initial,player):
        b_array.append(b)
    
    
    for x in b_array:
        minimax_values.append(minimax(x,0,False))
        maximum = max(minimax_values)
    
        
    for i in range(0,len(minimax_values)):
        if maximum == minimax_values[i]:
            return rec_array[i],b_array[i] 
        
print(solve(initial,player))