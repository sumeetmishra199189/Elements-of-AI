#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:01:25 2018
@author: varunmiranda
Citations:
https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
https://stackoverflow.com/questions/17870612/printing-a-two-dimensional-array-in-python
"""

import copy
import numpy as np

n = 3
x = n
player = "x"
#input = "...x..o.ox.oxxxooo"
input = "xoxoxoxoxoxoxoxoxo"

split = list(input)
turn = "o"
array = []

initial = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]
chunks = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]

def printable_board(chunks):    
    board = ('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in chunks]))
    print(board)

def opponent(player):
    if player == "x": 
        return "o"
    else: 
        return "x"


"Recommendation"

def successor(initial):
    for dr in range(-x,x+1):
        if dr > 0:
            value = drop_command(abs(dr),initial,dr)
            goal_state(value,n,dr)
        elif dr < 0:
            value = rotate_command(abs(dr),initial,dr) 
            goal_state(value,n,dr)
        dr = dr+1

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


def goal_state(chunks,n,dr):
    if(row_win(chunks,n) or col_win(chunks,n) or left_diag_win(chunks,n) or right_diag_win(chunks,n)):
        if dr < 0:
            print ('I would recommend rotating column '+str(abs(dr))+' and you will win')
        elif dr > 0:
            print ('I would recommend dropping a piece in column '+str(dr)+' and you will win')
        return True 

def row_win(chunks,n):
    turn = player
    enemy = opponent(player)
    row_c_p=0
    col_c_p=0
    ld_c_p=0
    rd_c_p=0
    
    row_c_o=0
    col_c_o=0
    ld_c_o=0
    rd_c_o=0
    
    for i in range(0,n):
        if (chunks[i][i] == turn):
            temp=chunks[i][i]
            count=0
            for j in range(0,n):
               # if chunks[i][j]!=temp:
               #     break
               if(chunks[i][j]==temp):
                    count+=1
    if(count==n):
                return True
    elif(count==n-1):
      for i in range(0,n):
          for j in range(0,n):
              if (i==0 and chunks[i][j]!=turn and chunks[i][j]!='.'):
                  if(chunks[n-1][j]==turn):
                       row_c_p=50
              elif(i!=0 and chunks[i][j]!=turn and chunks[i][j]!='.'):
                  if(chunks[i-1][j]==turn):
                       row_c_p=50
'''          
         for j in range(0,n):
            if (chunks[i][j]==chunks[i][i+1] and chunks[i][i+2]!='.'):
                if chunks[i][i]==chunks[n-1][i+2]: 
                   row_c_p=50           
            elif(chunks[i][i]==chunks[i][i+2] and chunks[i][i+1]!='.'):
                if chunks[i][i]==chunks[n-1][i+1]: 
                   row_c_p=50   
            elif(chunks[i][i+1]==chunks[i][i+2] and chunks[i][i]!='.'):       
                if chunks[i][i+1]==chunks[n-1][i]: 
                   row_c_p=50   
 '''           
    return False
            
def col_win(chunks,n):
    for j in range(0,n):
        if (chunks[j][j] == turn):
            temp=chunks[j][j]
            count=0
            for i in range(0,n):
                #if chunks[i][j]!=temp:
                 #   break
                elif(chunks[i][j]==temp):
                    count+=1
            if(count==n):
                return True
            elif(count==n-1):
                #for i in range(0,n):
                    for j in range(0,n):
                        i=0
                        if (chunks[i][j]==chunks[i+1][j]):
                            col_c_p=50
                            
                
    
    
    
    
    
    
    
    
    
    
    return False

                
def left_diag_win(chunks,n):
        i=0
        j=0
        if (chunks[i][j] == turn):
            temp=chunks[i][j]
            count=0
            while(i<n):
                if(chunks[i][j]==temp):
                    i+=1
                    j+=1
                    count+=1
                else:
                    break
            if(count==n):
                return True
            
            elif(count==n-1):
               while(i<n): 
                if (i=0 and chunks[i][j]!=turn):
                    if(chunks[n-1][j]==turn):
                        ld_c_p=50
                    elif(i!=0 and chunks[i][j]!=turn):
                        if(chunks[i-1][j]==turn):
                            d_c_p=50
                i+=1
                j+=1
                        
                
            
        return False
    
def right_diag_win(chunks,n):
        i=n-1
        j=0
        if (chunks[i][j] == turn):
            temp=chunks[i][j]
            count=0
            while(i>=0):
                if(chunks[i][j]==temp):
                    i-=1
                    j+=1
                    count+=1
                else:
                    break
            if(count==n):
                return True
            
            elif(count==n-1):
               while(i>=0): 
                if (i=0 and chunks[i][j]!=turn):
                    if(chunks[n-1][j]==turn):
                        ld_c_p=50
                    elif(i!=0 and chunks[i][j]!=turn):
                        if(chunks[i-1][j]==turn):
                            d_c_p=50
                i-=1
                j+=1
            
            
            
            
        return False















































successor(initial)