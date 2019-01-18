#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
"""
Created on Fri Oct 12 13:53:08 2018

@author: sumeetmishra
"""
import operator
def Roll_3_Dice(D1,D2,D3):
   
    closed={}
  
    A=[1,2,3,4,5,6]
    val_d1=0     #this will store all values by rerolling D1 
    val_d2=0     #this will store all values by rerolling D2 
    val_d3=0     #this will store all values by rerolling D3 
    fixed_d1=0   #this will store all values by rerolling D2,D3  
    fixed_d2=0   #this will store all values by rerolling D1,D3
    fixed_d3=0   #this will store all values by rerolling D1,D2
    all_value=0  #this will store all values by rerolling D1,D2,D3
    #if (D1 or D2 or D3 not in A):
     #   print('One or more invalid inputs,dies values should be between 1 to 6 ' )
      #  return
        
    if(D1 in A and D2 in A and D3 in A):
        if (D1==D2==D3):
            print('No Need To Reroll')
            return
        else:
            for i in range(1,7): #rerolling  D1 or D2 or D3(calculating total values of rerolling single dice)
                if(i==D2==D3):
                    val_d1=val_d1+25
                else:
                    val_d1=val_d1+D2+D3+i
                if(i==D1==D3):
                    val_d2=val_d2+25
                else:
                    val_d2=val_d2+D1+D3+i
                if(i==D1==D2):
                    val_d3=val_d3+25
                else:
                    val_d3=val_d3+D1+D2+i
            
                for j in range(1,7):# rerolling D1,D2 or D1,D3 or D2,D3 (calculating total values by rerolling 2 dies)
                    if(i==j==D3):
                        fixed_d3=fixed_d3+25
                    else:
                        fixed_d3=fixed_d3+i+j+D3
                    if(i==j==D2):
                        fixed_d2=fixed_d2+25               
                    else: 
                        fixed_d2=fixed_d2+i+j+D2
                    if(i==j==D1):
                        fixed_d1=fixed_d1+25               
                    else: 
                        fixed_d1=fixed_d1+i+j+D2
                    for k in range(1,7): #rerolling D1,D2,D3 (calculating total values by rerolling 4 dies)
                        if (i==j==k):
                            all_value=all_value+25
                        else:    
                            all_value=all_value+i+j+k
            e_val_d1=(val_d1/6)    #expected value of each combination by rerolling D1,divided by 6 as only 6 such combinations
            closed['D1']=e_val_d1            
            e_val_d2=(val_d2/6)    #expected value of each combination by rerolling D2,divided by 6 as only 6 such combinations
            closed['D2']=e_val_d2
            e_val_d3=(val_d3/6)    #expected value of each combination by rerolling D3,divided by 6 as only 6 such combinations
            closed['D3']=e_val_d3
            e_val_d1d2=(fixed_d3/36)  #expected value of each combination by rerolling D1 and D2,divided by 36 as only 36 such combinations 
            closed['D1','D2']=e_val_d1d2            
            e_val_d2d3=(fixed_d1/36)   #expected value of each combination by rerolling D2 and D3,divided by 36 as only 36 such combinations
            closed['D2','D3']=e_val_d2d3
            e_val_d3d1=(fixed_d2/36)     #expected value of each combination by rerolling D1 and D3,divided by 36 as only 36 such combinations
            closed['D3','D1']=e_val_d3d1
            e_val_d1d2d3=(all_value/216)  #expected value of each combination by rerolling D1 and D2 and D3,divided by 216 as only 216 such combinations
            closed['D1','D2','D3']=e_val_d1d2d3
            p=max(closed.items(), key=operator.itemgetter(1))[0]
            print('Reroll dice(dies) '+str(p))
            return                
    else:
        print('One or more invalid inputs,dies values should be between 1 to 6')
        return
        
    
    
    
Roll_3_Dice(1,6,3)              
                      
                      
                      
                      
