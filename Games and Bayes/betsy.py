#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:01:25 2018
@author: varunmiranda
Citations:
https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
https://stackoverflow.com/questions/17870612/printing-a-two-dimensional-array-in-python
"""

n = 3
input = "x..x..o.oxooxxxoxo"
split = list(input)
turn = "x"
drop_rotate = -2

col_chosen = abs(drop_rotate)

chunks = [split[i * n:(i + 1) * n] for i in range((len(split) + n - 1) // n )]

def printable_board(chunks):    
    board = ('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in chunks]))
    print(board)

"Drop Command"

def drop_command():
    for i in range(0,n+3):
        if chunks[i][col_chosen-1] != ".":
            chunks[i-1][col_chosen-1] = turn
            return chunks

"Rotate Command"

def rotate_command():
    for i in range(0,n+3):
        if chunks[i][col_chosen-1] != ".":
            temp = chunks[n+2][col_chosen-1]
            for j in range(n+2,i,-1):
                chunks[j][col_chosen-1] = chunks[j-1][col_chosen-1]
            chunks[i][col_chosen-1] = temp 
            return chunks
        
def row_win(chunks,n):
    for i in range(0,n):
        if (chunks[i][i] != '.'):
            temp=chunks[i][i]
            count=0
            for j in range(0,n):
                if chunks[i][j]!=temp:
                    break
                elif(chunks[i][j]==temp):
                    count+=1
            if(count==n):    
                #print('You Won')
                return True
            else:
                return False
def col_win(chunks,n):
    for j in range(0,n):
        if (chunks[j][j] != '.'):
            temp=chunks[j][j]
            count=0
            for i in range(0,n):
                if chunks[i][j]!=temp:
                    break
                elif(chunks[i][j]==temp):
                    count+=1
            if(count==n):    
                #print('You Won')
                return True
            else:
                return False
                
def left_diag_win(chunks,n):
        i=0
        j=0
        if (chunks[i][j] != '.'):
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
                #print('You Won')
                return True
            else:
                return False
def right_diag_win(chunks,n):
        i=n-1
        j=0
        if (chunks[i][j] != '.'):
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
                #print('You Won')
                return True
            else:
                return False
def goal_state(chunks,n):
    if(row_win(chunks,n) or col_win(chunks,n) or left_diag_win(chunks,n) or right_diag_win(chunks,n)):
        print('you Won')
            
if drop_rotate > 0 and drop_rotate <= n:
    drop_command()            
elif drop_rotate < 0 and drop_rotate >= -n:
    rotate_command()    

printable_board(chunks)

