# Program 2: Sudoku 
# Jenn Werth 

# Part 1: Specify the constraints for the 4x4 puzzle in Figure 4
# Each row, column, and box should have 1,2,3,4

# Row 1, Column 1, and Box 1 all have a 1 in it already 
# Row 2, Column 2, and Box 1 all have a 2 in it already 
# Row 3, Column 3, and Box 4 all have a 3 in it already 
# Row 4, Column 4, and Box 4 all have a 4 in it already 

# create a dictionary that holds the constraints 
from collections import deque
import copy
from sudoku_constraints import s_constraints
from flask import Flask, render_template, request


constraints = { ('C11', 'C12'): [[1,2], # row 1
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C11', 'C13'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C11','C14'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C12','C13'): [[1,2], 
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C12', 'C14'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C13', 'C14'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C11', 'C21'): [[1,2], # column 1
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C11', 'C31'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C11', 'C41'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C21', 'C31'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C21', 'C41'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C31', 'C41'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C12', 'C22'): [[1,2], # column 2
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C12', 'C32'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C12', 'C42'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C22', 'C32'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C22', 'C42'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C32', 'C42'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C21', 'C22'): [[1,2], # row 2
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C21', 'C23'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C21', 'C24'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C22', 'C23'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C22', 'C24'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C23', 'C24'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C13', 'C23'): [[1,2], # column 3
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C13', 'C33'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C13', 'C43'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C23', 'C33'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C23', 'C43'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C33', 'C43'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C31', 'C32'): [[1,2], # row 3
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C31', 'C33'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C31', 'C34'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C32', 'C33'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C32', 'C34'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C33', 'C34'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C14', 'C24'): [[1,2], # column 4
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C14', 'C34'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C14', 'C44'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C24', 'C34'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],  
                ('C24', 'C44'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],  
                ('C34', 'C44'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],  
                ('C41', 'C42'): [[1,2], # row 4
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],  
                ('C41', 'C43'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],  
                ('C41', 'C44'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C42', 'C43'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C42', 'C44'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C43', 'C44'): [[1,2],
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C12', 'C21'): [[1,2], # box 1
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C11', 'C22'): [[1,2], 
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C13', 'C24'): [[1,2], # box 2
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C14', 'C23'): [[1,2], 
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C31', 'C42'): [[1,2], # box 3
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C32', 'C41'): [[1,2], 
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C33', 'C44'): [[1,2], # box 4
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]],
                ('C34', 'C43'): [[1,2], 
                                [1,3],
                                [1,4],
                                [2,1],
                                [2,3],
                                [2,4],
                                [3,1],
                                [3,2],
                                [3,4],
                                [4,1],
                                [4,2],
                                [4,3]]}

# CSP as shown in figure 3 on writeup 
# lists the variables and the corresponding constraints
CSP_fig4 = {'variables': {'C11': [1],
                         'C12': [1,2,3,4],
                         'C13': [1,2,3,4],
                         'C14': [1,2,3,4],
                         'C21': [1,2,3,4],
                         'C22': [2],
                         'C23': [1,2,3,4],
                         'C24': [1,2,3,4],
                         'C31': [1,2,3,4],
                         'C32': [1,2,3,4],
                         'C33': [3],
                         'C34': [1,2,3,4],
                         'C41': [1,2,3,4],
                         'C42': [1,2,3,4],
                         'C43': [1,2,3,4],
                         'C44': [4]},
            'constraints': constraints}

# helper function that converts a 2D array into a 9x9 CSP to represent a sudoku board 
def to_CSP(board):
    variables = {}
    for row in range(len(board)):
        for column in range(len(board[row])):
            index = 'C'+str(row+1)+str(column+1)
            if board[row][column] == None:
                variables[index] = [1,2,3,4,5,6,7,8,9]
            else:
                variables[index] = [board[row][column]]
    
    # assignments = solution (in theory at the end) 
    # ordered = order to get to solution 
    # boards = this is used for the html stuff for our app later on 
    csp = {'variables': variables, 'constraints': s_constraints, 'assignments': [], 'ordered': [], 'boards': []}
    return csp

# PART 2: Write a function revise that takes a CSP and the names of two variables as input
# Modifies the CSP, removing any value in the first variable’s domain 
# where there isn’t a corresponding value in the other variable’s domain that satisfies the constraint between the variables
# The function should return a boolean indicating whether or not any values were removed

# only removes from one of the domains 
def revise(CSP, var1, var2):

    # Determine the domains for the two variables defined 
    domain1 = CSP['variables'][var1]
    domain2 = CSP['variables'][var2]
    changed = False
    #assignment = None

    # Find our constraints between the two variables
    # note: we must determine which order the keys are within our dictionary
    constraints = []
    if (var1, var2) in CSP['constraints']:
        constraints =  CSP['constraints'][(var1, var2)]
    elif (var2, var1) in CSP['constraints']:
        constraints =  CSP['constraints'][(var2, var1)]
    else:
        # in case we are given invalid keys or two boxes are unrelated
        #print("invalid key comparison")
        return False

    # look at all values in first domain 
    # check if there's a value in the second cell's domain that satisfies constraint between them 
    consistent = []
    for d1 in domain1: 
        for d2 in domain2: 
            if [d1, d2] in constraints and d1 not in consistent:
                consistent.append(d1)
            """
            else:
                print(d1, d2, "are inconsistent")
                print("from", var1, var2)
            """
    #print("final, ", consistent)
    if consistent != domain1: 
        changed = True

    # reassign the domain to variable 1 with the removed inconsistencies 
    CSP['variables'][var1] = consistent

    return changed


# PART 3: Implement AC-3 as a function
# Takes as input a CSP and modifies it such that any inconsistent values across all domains are removed
# Return a boolean indicating whether or not all variables have at least on value left in their domains
def AC3(CSP):
    #print("AC3 FUNCTION CALL")

    hasVars = True
    edges = deque()

    # create an initial queue for our AC3 algorithm
    # should contains (var1, var2) and (var2, var1) keys 
    for constraint1 in CSP['variables'].keys():
        for constraint2 in CSP['variables'].keys():
            edges.append((constraint1, constraint2))

    # loop while there are still edges in the queue 
    while len(edges) > 0: 
        (Xi, Xj) = edges.popleft()

        # check if we changed anything
        #print("revising change from ", Xi, Xj)
        change = revise(CSP, Xi, Xj)

        # if we changed Xi's domain, then we must recheck all (Xk, Xi) 
        if change: 
            #assignments[assigned[0]] = assigned[1]
            if len(CSP['variables'][Xi]) == 0:
                #print("variable ", Xi, " now has empty domain because of ", Xj)
                #print(CSP['variables'])
                return False    # if there is a domain with an empty set of variables

            # add all (Xk, Xi) keys back to the queue so we can check them again 
            for Xk in CSP['variables'].keys():
                edges.append((Xk, Xi))
    return hasVars

# PART 4: minimum-remaining-values
# Takes a CSP and a set of variable assignments as input
# Returns the variable with the fewest values in its domain among the unassigned variables in the CSP.
def minimum_remaining_values(CSP, vars):
    keys = []
    for var in vars:
        keys.append(var[0])

    for variable in CSP['variables']:
        #print("variable ", variable)
        if variable not in keys: 
            min = variable
            break

            
    # vars represents the assigned variables, we want to check the unassigned variables 
    for var in CSP['variables']:
        if len(CSP['variables'][var]) < len(CSP['variables'][min]) and var not in keys:
            min = var

    return min

# PART 5: backtracking search 

# HELPER FUNCTIONS 
# takes a CSP and a set of already assigned variables and returns an array of the newly assigned variables
def find_assigned(CSP, assignments):
    newly_assigned = []
    for var in CSP['variables']:
        if len(CSP['variables'][var]) == 1 and (var,CSP['variables'][var]) not in assignments:
            newly_assigned.append((var,CSP['variables'][var]))
    return newly_assigned

def solved(CSP):
    for var in CSP['variables']:
        #print(len(curr_board['variables'][var]))
        if len(CSP['variables'][var]) > 1:   # break the second we find a variable with more than one arg in domain 
            return False
    return True 

# BACKTRACKING 
def backtracking(CSP):
    hold = copy.deepcopy(CSP)
    return rec_backtracking(CSP, hold)

def rec_backtracking(CSP, original):

    # find the unassigned variables
    unassigned = copy.deepcopy(CSP['variables'])

    # determine which variables are already assigned
    assigned = find_assigned(CSP, [])
    # keep track of the state of boards after each assignment
    # this is later used in the html stuff 
    CSP['boards'].append(assigned) 

    # find the assigned variables that are not already in ordered
    # these come from the AC3 elimination 
    # add these to the ordered key so that we can also track when these get assigned
    for i in range(len(assigned)):
        if assigned[i][0] not in CSP['ordered']:
            CSP['ordered'].append(assigned[i][0])
    
    # find the unassigned variables
    for var in CSP['variables']:
        if len(CSP['variables'][var]) == 1:
            unassigned.pop(var)


    # if everything has been assigned then we are good to go 
    if len(unassigned) == 0:
        print("******************* SOLVED PUZZLE **************************")
        print(assigned)
        CSP['assignments'] = assigned
        print("order of assignments: \n", CSP['ordered'])
        return CSP

    next_var = minimum_remaining_values(CSP, assigned)
    unassigned.pop(next_var)

    index = 0
    while index < len(CSP['variables'][next_var]):

        #print("CURRENT VARIABLES", CSP['variables'])

        guess = CSP['variables'][next_var][index]
        CSP['ordered'].append(next_var)

        #print("CURRENT VARIABLE: ", next_var, CSP['variables'][next_var])
        print("ASSIGNMENT:", next_var, "=", guess)
        #print("UNASSIGNED:", unassigned)

        # hold a temporary domain in case we guess wrong 
        domain = CSP['variables'][next_var]
        # then assign the domain of the next var to be our guess
        CSP['variables'][next_var] = [guess]

        # Test on a copy of our CSP as well in case we go wrong 
        hold = copy.deepcopy(CSP)
        #print("hold variables:", hold['variables'])

        if (AC3(hold) == False):
            # if this is inconsistent, then remove it as a possibility from the domain 
            domain.remove(guess)
            # set this domain to be our CSP's domain 
            CSP['variables'][next_var] = domain
            CSP['ordered'].remove(next_var)
            #unassigned[next_var] = domain
        else:
            # again we test on a copy
            # we want this BEFORE our AC3 was done on it 
            
            hold = copy.deepcopy(CSP)
            consistent = rec_backtracking(hold, AC3(hold))
            if consistent != False:
                #CSP['boards'].append(assigned)
                return consistent
            else:
                domain.remove(guess)
                #CSP['ordered'].remove(next_var)
                CSP['variables'][next_var] = domain
                #CSP['failed'].append((next_var,guess))
                
                if assigned in CSP['boards']:
                    CSP['boards'].remove(assigned)
    return False


# PART 6: a web-based visualization of your backtracking search
# allowthe user to specify a Sudoku puzzle
# show the board state at each step of the problem

# HELPTER FUNCTIONS FOR THIS PART 

# We need to be able to convert from HTML to CSP 
html_to_csp = {'variables': {},
                'constraints': s_constraints,
                'assignments': [],
                'ordered': [],
                'boards': []}

# helper function for posting to web developer 
# takes a CSP and puts it in format we can use for the web app
def html_steps(CSP):
    # create an array to hold our steps 
    steps = []
    # create a dictionary of the final assignments
    steps_dict = {  'C11': [],
                    'C12': [],
                    'C13': [],
                    'C14': [],
                    'C15': [],
                    'C16': [],
                    'C17': [],
                    'C18': [],
                    'C19': [],
                    'C21': [],
                    'C22': [],
                    'C23': [],
                    'C24': [],
                    'C25': [],
                    'C26': [],
                    'C27': [],
                    'C28': [],
                    'C29': [],
                    'C31': [],
                    'C32': [],
                    'C33': [],
                    'C34': [],
                    'C35': [],
                    'C36': [],
                    'C37': [],
                    'C38': [],
                    'C39': [],
                    'C41': [],
                    'C42': [],
                    'C43': [],
                    'C44': [],
                    'C45': [],
                    'C46': [],
                    'C47': [],
                    'C48': [],
                    'C49': [],
                    'C51': [],
                    'C52': [],
                    'C53': [],
                    'C54': [],
                    'C55': [],
                    'C56': [],
                    'C57': [],
                    'C58': [],
                    'C59': [],
                    'C61': [],
                    'C62': [],
                    'C63': [],
                    'C64': [],
                    'C65': [],
                    'C66': [],
                    'C67': [],
                    'C68': [],
                    'C69': [],
                    'C71': [],
                    'C72': [],
                    'C73': [],
                    'C74': [],
                    'C75': [],
                    'C76': [],
                    'C77': [],
                    'C78': [],
                    'C79': [],
                    'C81': [],
                    'C82': [],
                    'C83': [],
                    'C84': [],
                    'C85': [],
                    'C86': [],
                    'C87': [],
                    'C88': [],
                    'C89': [],
                    'C91': [],
                    'C92': [],
                    'C93': [],
                    'C94': [],
                    'C95': [],
                    'C96': [],
                    'C97': [],
                    'C98': [],
                    'C99': []}
    step = steps_dict
    index = 0

    for board in CSP['boards']:
        # construct the original board 
        if index == 0:
            for var in board:
                #print('var[0] = ', var[0])
                step[var[0]] = CSP['variables'][var[0]]
            steps.append(step)
            index += 1
        else:
            for var in board:
                if len(steps) > 0:
                    step = copy.deepcopy(steps[len(steps)-1])
                else:
                    step = steps_dict
                step[var[0]] = CSP['variables'][var[0]]
                steps.append(step)

    # returns an array of dictionaries 
    return steps

puzzle4 = [[None, None, None, None, 9, None, None, 7, 5], 
                [None, None, 1, 2, None, None, None, None, None],
                [None, 7, None, None, None, None, 1, 8, None], 
                [3, None, None, 6, None, None, 9, None, None],
                [1, None, None, None, 5, None, None, None, 4],
                [None, None, 6, None, None, 2, None, None, 3],
                [None, 3, 2, None, None, None, None, 4, None],
                [None, None, None, None, None, 6, 5, None, None],
                [7, 9, None, None, 1, None, None, None, None]]
csp_p4 = to_CSP(puzzle4)

# FLASK 
# web visualizer 
app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def sudoku():
    if request.method == "POST":
        temp_constraints = {}
        for key in request.form:
            if request.form[key] != "":
                temp_constraints.update({key: [int(request.form[key])]})
            else:
                temp_constraints.update({key: [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        html_to_csp['variables'] = temp_constraints
        solution = dict(backtracking(html_to_csp))
        
        step_array = html_steps(solution)
        return render_template('solved.html', solutions = step_array, len = len(step_array))
    return render_template('sudoku.html')


def main():
    # run the app 
    app.run()

    puzzle1 = [[7,None,None,4,None,None,None,8,6], 
                [None, 5, 1, None, 8, None, 4, None, None],
                [None, 4, None, 3, None, 7, None, 9, None],
                [3, None, 9, None, None, 6, 1, None, None], 
                [None, None, None, None, 2, None, None, None],
                [None, None, 4, 9, None, None, 7, None, 8],
                [None, 8, None, 1, None, 2, None, 6, None],
                [None, None, 6, None, 5, None, 9, 1, None], 
                [2, 1, None, None, None, 3, None, None, 5]]

    puzzle4 = [[None, None, None, None, 9, None, None, 7, 5], 
                [None, None, 1, 2, None, None, None, None, None],
                [None, 7, None, None, None, None, 1, 8, None], 
                [3, None, None, 6, None, None, 9, None, None],
                [1, None, None, None, 5, None, None, None, 4],
                [None, None, 6, None, None, 2, None, None, 3],
                [None, 3, 2, None, None, None, None, 4, None],
                [None, None, None, None, None, 6, 5, None, None],
                [7, 9, None, None, 1, None, None, None, None]]


    csp_p1 = to_CSP(puzzle1)
    csp_p4 = to_CSP(puzzle4)

    csp_p4['order'] =[]
    csp_p4['assignments'] =[]
    csp_p4['failed'] =[]

    """
    solution = backtracking(csp_p4)
    print(solution['boards'])
    html = html_steps(solution)
    print("BOARD 1")
    print(html[0])

    print("BOARD 2")
    print(html[1])
    """

    #print(html[0])

    #asgmt = find_assigned(CSP_fig4, [])
    #solution = backtracking(csp_p4)
    #print(solution.type)
    #print(solution['ordered'])
    #print(solution['assignments'])

# HARD CODING FOR TESTING PURPOSES 

    """
    # call AC3 on figure 4 
    test = AC3(CSP_fig4)
    print(test)
    print(CSP_fig4['variables'])    # this eliminates inconsistencies


    # pick a random variable for minimum remaining domain 
    CSP_fig4['variables']['C12'] = [3]
    # run AC3 on that new assignment 
    test = AC3(CSP_fig4)
    print(test)
    print(CSP_fig4['variables'])

    # check if solved -- it is.
    print(solved(CSP_fig4))
    
    
    CSP_fig4['assignments'] = []
    CSP_fig4['failed'] = []
    """

    #solution = backtracking_search(CSP_fig4)

    # printing failure 
    #print(solution)

    #print(csp_p1['variables'])
    #revise(csp_p1, 'C12', 'C11')

    #AC3(csp_p1)
    #print(backtracking_search(csp_p1))
    #steps = backtracking_search(csp_p1)
    #print("------------------ STEPS ---------------")
    #print(steps)

    proof_of_concept = [[1, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None]]
    p_csp = to_CSP(proof_of_concept)

    #revise(p_csp, 'C12', 'C13')
    #print(p_csp['variables'])

    """
    csp_proof = to_CSP(proof_of_concept)
    test1 = backtracking_search(csp_proof)
    test2 = backtracking_search(csp_p1)
    print(test1)
    
    for step in test2:
        print("Assignment: ", step)
    """


if __name__ == "__main__":
   main()
