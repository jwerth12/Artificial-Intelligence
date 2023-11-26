from collections import deque
import math
from queue import PriorityQueue
from copy import deepcopy

# profiler 
import cProfile
import re

# find empty space helper function
def find_empty(board: list):
    n_row = 0 # null row
    n_col = 0 # null column
    for row in range(len(board)):
        for column in range(len(board[row])):
            if board[row][column] == None:
                n_row = row
                n_col = column
    return [n_row, n_col]

# formatting helper function to display the boards
def format(board: list):
    for row in board:
        print(row)

# Part 1: 
# possible-actions: takes a board as input 
# and outputs a list of all actions possible on the given board
# This assumes that we are moving the number, not the empty space
def possible_actions(board: list):

    # find the null space
    n_row = find_empty(board)[0]
    n_col = find_empty(board)[1]

    # the spaces adjacent to the null space can perform actions 
    # components of an action: 
    # direction (up, down, left, right), row, and column (of tile)
    actions = []
    if n_row > 0:
        actions.append([n_row-1, n_col, "down"])
    if n_row < len(board)-1:
        actions.append([n_row+1, n_col, "up"])
    if n_col > 0:
        actions.append([n_row, n_col-1, "right"])
    if n_col < len(board[n_row])-1:
        actions.append([n_row, n_col+1, "left"])

    # print(actions)
    return actions

# Part 2: 
# Result: takes as input an action and a board and outputs the new board that will result 
def result(board: list, action: list):
    #print("BOARD: ")
    #format(board)
    #print("ACTION: ", action)
    newboard = []
    for spot in board:
        newboard.append(spot)
    empty = find_empty(board)
    n_row = empty[0]
    n_col = empty[1]
    newboard[n_row][n_col] = board[action[0]][action[1]]
    newboard[action[0]][action[1]] = None
    return newboard

# Part 3: 
# Expand: takes a board as input, and outputs a list of all states that can be
# reached in one Action from the given state
# Returns as a dictionary that contains the new board, the parent board, 
# the action it took to get there, and the path cost
def expand(board: list):
    actions = possible_actions(board)
    expanded = []
    current = deepcopy(board)
    for action in actions:
        output = result(current, action)
        current = deepcopy(board)
        #print("RESULT FROM ACTION", actions[i])
        #format(output)
        expanded.append(output)
    #print_expanded(expanded)
    return expanded


# Part 4:
# Iterative deepening search 
# takes in an initial board and a goal board and outputs the the states of each step to get to the goal

# Node class to help track path cost and parent nodes for a given board
# a node has the current state (a board), a parent (a board), and a path_cost (which equals 1+parent.pathcost)
class Node:
    def __init__(self, board, parent, path_cost):
        self.board = board
        self.parent = parent
        self.path_cost = path_cost

# iterative deepening function which calls depth limited search 
def iterative_deep(initial_board, goal_board):
    max_depth = 0
    while True:
        solution = dls(initial_board, goal_board, max_depth)
        if solution != False:
            steps = []
            curr = solution
            while curr != None:
                steps.append(curr.board)
                curr = curr.parent
            steps.reverse()
            return steps
        max_depth += 1

# depth limited search to implement iterative deepening 
# takes in an initial board and a goal baord and a max depth 
# ouputs the NODE of the goal board (which in turn contains all the steps it took to get there)
def dls(initial_board, goal_board, max_depth):
    frontier = deque()
    result = False

    start = Node(initial_board, None, 0)
    frontier.appendleft(start) # appending left and popping left makes this a lifo 

    # while not empty, iterate
    while len(frontier) > 0:
        curr = frontier.popleft()

        # FOR TESTING PURPOSES 
        """
        print("CURRENT BOARD")
        print(curr.board)
        print("GOAL BOARD")
        print(goal_board)
        """

        # check if we reached the goal 
        if curr.board == goal_board:
            #print("REACHED SOLUTION")
            return curr

        # check if we reached max depth (max_depth can be represented by path cost of the node because we increment every time we expand)
        #print("current depth", curr.path_cost)
        if curr.path_cost < max_depth:

        # check if we are entering a cycle

            if not isCycle(curr):
                # if it is not a cycle, then expand on that node to explore all children
                next_moves = expand(curr.board)
                for move in next_moves:
                    child = Node(move, curr, curr.path_cost+1)
                    # create child node that contains curr as parent, path cost = parent.pathh_cost +1 and put THAT into the queue 

                    frontier.appendleft(child)
    return result


# isCycle helper function for iterative deepening 
# returns True if we find a cycle, False if do not find a cycle from current board up to parent 
# cycles occur if we come back to a baord state we already reached before 
def isCycle(board: Node):
    reached = []
    curr = deepcopy(board)
    # follow path back up to parent to track what nodes we've reached 
    
    while curr.parent != None:
        curr = curr.parent
        reached.append(curr.board)

    # check if the current board we are at has been reached before (aka there's a cycle)
    if board.board in reached:
        #print("CYCLE!!")
        return True
    
    #print("NOT A CYCLE")
    return False

# Part 5: Breadth first search 
# takes in an initial board and a goal board 
# outputs the states of the steps it took to get from intial to goal 
def bfs(initial_board, goal_board):
    start = Node(initial_board, None, 0) # create starting node 
    if initial_board == goal_board:
        return start

    frontier = deque()  # create a first in first out queue 
    frontier.append(start)

    reached = []
    reached.append(start.board)  # store the initial board in a reached array 

    while len(frontier) > 0:
        curr = frontier.popleft()   # note: we are doing a fifo queue this time 
        
        next_steps = expand(curr.board)
        
        for step in next_steps:
            child = Node(step, curr, curr.path_cost+1)

            # if we find the solution
            if step == goal_board:
                steps = []
                while child != None:
                    steps.append(child.board)
                    child = child.parent
                steps.reverse()
                return steps

            # if we are not at the solution but the child has not yet been explored, add it to the end of the queue
            if step not in reached:
                reached.append(step)
                frontier.append(child)
    return False

# PART 6
# A * Seach: takes in an initial board, a goal board, and a heuristic function 
# the heuristic function is used to determine priority 
# in this case we assume path cost is determined by the heuristic 
def a_star(initial_board, goal_board, heuristic):

    
    # create a priority queue ordered by f with start node as element
    pq = PriorityQueue()    
    start = Node(initial_board, None, heuristic(initial_board)+1)

    start_key = createKey(start)
    pq.put((0, start_key))  

    # create a lookup dictionary for reached nodes 
    # put in one entry for key of initial problem and the value of node 
    
    reached = {}
    reached[start_key] = start

    # iterate as long as the priority queue is not empty 
    while not pq.empty():
        prio, curr = pq.get()
        #curr_key = createKey(curr)

        if reached[curr].board == goal_board:
            
            #print("REACHED")

            steps = []
            hold = reached[curr]
            while hold != None:
                steps.append(hold.board)
                hold = hold.parent
            steps.reverse()
            return steps
        
        next_steps = expand(reached[curr].board)
        for step in next_steps:
            # the priority is determined by the heuristic function 
            # add it to the parent path cost because options with more steps are less ideal 
            priority = heuristic(step) + reached[curr].path_cost
            child = Node(step, reached[curr], priority)
            child_key = createKey(child)
            if child_key not in reached or child.path_cost < reached[child_key].path_cost:
                reached[child_key] = child
                pq.put((priority, child_key))
    return False

# helper function to create keys for the lookup table in A* search 
# takes in a node and returns a stringified version of the board 
def createKey(node):
    hashString = ""
    for i in range(len(node.board)):
        for j in range(len(node.board)):
            if(node.board[i][j] == "nil"):
                hashString += "0"
            hashString += str(node.board[i][j])
    return hashString

# Heuristic functions

# heuristic function for misplaced tiles (hamming distance)
# hamming distance = total number of misplaced tiles 
def misplaced_tiles(board):
    total = 0
    for row in range(len(board)):
        for tile in range(len(board[row])):
            if board[row][tile] != None:
                if board[row][tile] != row*len(board[row])+1:
                    total += 1
    return total

# heuristic function for manhatten distance 
# h(n) = Sum( distance(tile, correct_position) )
def manhatten_dist(board):
    dist = 0
    for row in range(len(board)):
        for tile in range(len(board[row])):
            if board[row][tile] != None:
                correct_x = board[row][tile] / len(board)
                correct_x = math.floor(correct_x)
                correct_y = board[row][tile] / len(board[row])
                correct_y = math.floor(correct_y)
                dist += distance(row, tile, correct_x, correct_y)
    return dist

def distance(tile_x, tile_y, correct_x, correct_y):
    x_dist = abs(correct_x - tile_x)
    y_dist = abs(correct_y - tile_y)
    return x_dist + y_dist

def main():
    
    print("************************ TESTING ITERATIVE DEEPENING ***************************")
    print(" ------- PUZZLE 0 --------")

    puzzle0 = [[3, 1, 2], [7, None, 5], [4, 6, 8]]

    # for testing purposes i have each step board that puzzle 0 takes to get to final goal 
    goal1 = [[3,1,2], [None, 7,5], [4,6,8]]
    goal2 = [[3,1,2],[4,7,5,],[None,6,8]]
    goal3 = [[3,1,2], [4,7,5], [6,None,8]]
    goal4 = [[3,1,2], [4,None,5], [6,7,8]]
    goal5 = [[3,1,2], [None,4,5], [6,7,8]]

    # this one is the final goal 
    goal6 = [[None,1,2], [3,4,5], [6,7,8]]

    
    test = iterative_deep(puzzle0, goal6)
    for board in test:
        print("--- STEP ---")
        format(board)

    # this should take 26 steps to get to (from book)
    # THIS TAKES A LONG TIME 
    puzzle1 = [[7,2,4], [5, None, 6], [8,3,1]]
    #test1 = dls(puzzle1, goal6, 26)

    # TESTING FOR BREADTH FIRST 
    print("*********************TESTING BREADTH FIRST ***********************")
    print(" ------- PUZZLE 0 --------")
    format(puzzle0 ) # just making sure this hasn't changed because python is weird with dynamic stuff 
    print("---------------------------")
    bfsTest = bfs(puzzle0, goal6)
    print("**** SOLUTION ******")
    for board in bfsTest:
        print("--- STEP ---")
        format(board)
    """
    print("testing now with puzzle 1")
    format(puzzle1)
    bfsTest1 = bfs(puzzle1, goal6)
    print("**** SOLUTION ******")
    for board in bfsTest1:
        print("STEP")
        format(board)
    """

    print("*********************** TESTING A* SEARCH ***************************")
    print(" ------- PUZZLE 0 --------")

    # testing with a heuristic that just returns 0 
    def ret_zero(baord):
        return 0

    a_star_test = a_star(puzzle0, goal6, ret_zero)
    for board in a_star_test:
        print("--- STEP ---")
        format(board)

    # Testing using the heuristic functions
    print(" ********** HEURISTIC MISPLACED TILES *************** ")

    # test with puzzle 0 
    print(" ------- PUZZLE 0 --------")
    a_star_test = a_star(puzzle0, goal6, misplaced_tiles)
    for board in a_star_test:
        print("--- STEP ---")
        format(board)
    
    # test with puzzle 1
    print(" ------- PUZZLE 1 --------")
    a_star_test = a_star(puzzle1, goal6, misplaced_tiles)
    for board in a_star_test:
        print("--- STEP ---")
        format(board)

    print(" *********** HEURISTIC MANHATTEN DIST **************** ")

    # test with puzzle 0 
    print(" ------- PUZZLE 0 --------")
    a_star_test = a_star(puzzle0, goal6, manhatten_dist)
    for board in a_star_test:
        print("--- STEP ---")
        format(board)   
    
    # test with puzzle 1 
    print(" -------- PUZZLE 1 ---------")
    a_star_test = a_star(puzzle1, goal6, manhatten_dist)
    for board in a_star_test:
        print(" --- STEP ---")
        format(board)
    
if __name__ == "__main__":
   main()