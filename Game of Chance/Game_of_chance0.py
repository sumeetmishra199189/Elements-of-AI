#!/usr/bin/env python3
import sys
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:31:39 2018

@author: sumeetmishra
"""

def Roll_3_Dice(D1,D2,D3):
   
    e_value=3.5                #expected value of a dice is 3.5
    A=[1,2,3,4,5,6]

    if D1 in A:
        if D2 in A:
            if D3 in A:
                if (D1>e_value and D2>e_value and D3>e_value):
                    print('No need to reroll')
                elif(D1>e_value and D2>e_value and D3<e_value): 
                    print('Reroll dice 3')
                elif (D1>e_value and D2<e_value and D3>e_value):
                    print('Reroll dice 2')
                elif (D1<e_value and D2>e_value and D3>e_value):
                    print('Reroll dice 1')
                elif (D1>e_value and D2<e_value and D3<e_value):
                    print('Reroll dice 2 and dice 3')
                elif (D1<e_value and D2<e_value and D3>e_value):
                    print('Reroll dice 1 and dice 2')
                elif (D1<e_value and D2>e_value and D3<e_value):
                    print('Reroll dice 1 and dice 3')
                else:
                    print('Reroll all the dices')
                
            else:
                print('Invalid input for 3rd Dice')
                return False
        else:
            print('Invalid input for 2nd Dice')
            return False  
    else:
        print('Invalid input for 1st Dice')
        return False        
    
    D1 =(sys.argv[1])
    D2 =(sys.argv[2])
    D3 =(sys.argv[3])
print(" Input Dices are" +D1+'\n'+D2+'\n'+D3 + "\n\nLooking for solution...\n")
Roll_3_Dice(D1,D2,D3)
        