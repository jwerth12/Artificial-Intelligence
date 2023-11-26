# Jenn Werth 
# Program 3: Hexapawn

from math import exp
import numpy as np

# PART 1: Implement a a Formalization of Hexapawn
# a state description as well as functions TOMOVE(s), ACTIONS(s), RESULT(s, a), IS-TERMINAL(s), and UTILITY(s)

# STATE DESCRIPTION 
# A state contains a board (with the location of pieces)
#   a board is a 2D array represented as a vector of 10 values
#   the first value specifies whose turn it is, 
#   and the other 9 represent the board in row major order where each cell will be: 
#       0 – an empty square 
#       1 – a white pawn 
#       -1 – a black pawn
# Who's turn it is:
#   0: black pawn's turn 
#   1: white pawn's turn 

# Example State:
initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]   # taken from figure 1 on writeup
# intial_state[0]: whose turn it is
# initial_state[1:4]: the top row – intiially all black pawns
# intial_state[4:7]: middle row – initially all empty spaces 
# intiial_state[7:9]: third row – initially all white pawns

# helper function takes in a state and outputs a 2d array rep of board
def to_board(state):
    r1 = state[1:4]
    r2 = state[4:7]
    r3 = state[7:10]
    return [r1, r2, r3]

# helper function takes in a 2d array rep or board and outputs a state as defined above 
def to_state(turn, board):
    state = [turn]
    for row in board:
        for space in row:
            state.append(space)
    return state

# TOMOVE(s)
# returns the player whose turn it is in state s 
def to_move(state):
    return state[0]

# ACTIONS(s) 
# takes in a state and outputs a list of actions that can be done on that board
#   Actions are of the form [row, column, action, row, column]
#       where the first row and colum tell us where the pawn we are moving is 
#       and if the action is 'capture' the second row and column tell us where the pawn we are capturing is
# white pawns can move up one square or can capture a black pawn diagonally up one square, 
# black pawns can move down one square or can capture a white pawn diagonally down one square
def actions(state):
    turn = state[0] # 0 = black turn, 1 = white turn 
    possible_actions = []   # create a list of possible actions that can be taken 
    
    # simplify state into a 2d array for iterative purposes 
    board = to_board(state)

    # Handle black turn 
    if turn == 0:
        for r in range(3):
            for c in range(3):
                if board[r][c] == -1:
                    if r < 2:
                        if board[r+1][c] == 0:
                            action = [r, c, 'down']
                            possible_actions.append(action)
                        if c+1 <= 2 and board[r+1][c+1] == 1:
                            action = [r, c, 'capture', r+1, c+1]
                            possible_actions.append(action)
                        if c-1 >= 0 and board[r+1][c-1] == 1:
                            action = [r, c, 'capture', r+1, c-1]
                            possible_actions.append(action)
    # Handle white turn 
    if turn == 1:
        for r in range(3):
            for c in range(3):
                if board[r][c] == 1:
                    if r > 0:
                        if board[r-1][c] == 0:
                            action = [r, c, 'up']
                            possible_actions.append(action)
                        if c+1 <= 2 and board[r-1][c+1] == -1:
                            action = [r, c, 'capture', r-1, c+1]
                            possible_actions.append(action)
                        if c-1 >= 0 and board[r-1][c-1] == -1:
                            action = [r, c, 'capture', r-1, c-1]
                            possible_actions.append(action)

    return possible_actions

# RESULT(s, a)
# takes a current state and an action, and outputs the resulting state 
# actions are in formated as arrays 
# NOTE: this does not actually modify the inputted board 
def result(state, action):
    # we must switch turns 
    player = state[0]
    if player == 0:
        next_player = 1
    else:
        next_player = 0
    
    board = to_board(state)
    r_pawn = action[0]
    c_pawn = action[1]

    if action[2] == 'up':
        board[r_pawn-1][c_pawn] = board[r_pawn][c_pawn]
    elif action[2] == 'down':
        board[r_pawn+1][c_pawn] = board[r_pawn][c_pawn]
    elif action[2] == 'capture':
        board[action[3]][action[4]] = 0
    board[r_pawn][c_pawn] = 0
    
    # TESTING PURPOSES
    """
    print("RESULTING BOARD FROM", action)
    for row in board:
        print(row)
    """

    result_state = to_state(next_player, board)
    return result_state

# IS_TERMINAL 
# takes in a state and returns a boolean value representing if the game has finished 
# The goal of each player is to either get one of their pawns to the other end of the board, 
# or to make it such that their opponent is stuck on their next move.
# NOTE: as is this does not tell us who won 
def is_terminal(state):
    # look to see if a white pawn has made it onto the first row
    for i in range(1,4):
        if state[i] == 1:
            #print("white won")
            return True
    # look at the last row to see if a black pawn made it 
    for i in range(7, 10):
        if state[i] == -1:
            #print("black won")
            return True

    # check if other opponent stuck on move 
    possible_actions = actions(state)
    if len(possible_actions) == 0:
        """
        if (state[0] == 1):
            # this means the white pawn is stuck with no moves
            print("black won")
        else:
            # this means the black pawn is stuck with no moves 
            print("white won")
        """
        return True
    # otherwise the game is still active 
    return False

# UTILITY(s)
# give the utility of MAX (white) and know that MIN’s (black) utility is just the negation of that.
#   defines the final numeric value to player p when the game ends in terminal state
#   win = 1, loss = 0 (zero sum game so we only care about wins and losses)
# Returns: the utility of the state
def utility(state):
    # utility is found once we have hit a terminal state
    if is_terminal(state):
        # if it is black pawn's turn, and we have hit a terminal state, then white has won because game ends the second someone wins 
        if state[0] == 0:
            return 1   # MAX win
        else:
            return -1    # MIN wins 
    # if we are not in a termianl state then our utility is 0 
    return 0



# PART 2: minimax search to build a policy table
# builds a policy table for a formalized game (like in part 1)
# For each state, the policy should include the value of the game 
# as well as every action that achieves that value.

# minimax search 
# finds the optiimal next move given a state 
# returns the next best move
def minimax_search(state):
    player = to_move(state)
    minimax_val, move = max_value(state, state)
    return move

# returns a utility, move pair 
def max_value(original, curr_state):
    if is_terminal(curr_state):
        return utility(curr_state), None
    possible_actions = actions(curr_state)
    v = -999999999
    for action in possible_actions:
        #print("action:", action)
        v2, a2 = min_value(curr_state, result(curr_state, action))
        # this if statement only gets triggered if the white pawn (MAX) wins the game 
        if v2 > v:
            v = v2
            move = action
    return v, move

# returns a utility, move pair 
def min_value(original, curr_state):
    if is_terminal(curr_state):
        return utility(curr_state), None
    v = 99999999
    possible_actions = actions(curr_state)
    for action in possible_actions:
        #print("action:", action)
        v2, a2 = max_value(curr_state, result(curr_state, action))
        # this if statement only gets triggered if the black pawn (MIN) wins the game
        # otherwise v2 will not have a value 
        if v2 < v:
            #print("found a new min value")
            v = v2
            move = action 
    return v, move

# build policy table 
# policy – a mapping from every possible state to the best move in that state.
# table constructed using retrograde minimax search 
# returns:
#   a dictionary where the key is the state of the game
#       and the value is contains the current value of the game, and the next best move 
#       this is in the form (curr_value, next_best_move)
def build_policy(state):
    table = {}
    # generate all the next possible boards we can create from this state 
    all = all_boards(state, [])
    for board in all:
        move = minimax_search(board)
        curr_val = utility(board)
        val_achieved = curr_val
        if move != None:
            val_achieved = utility(result(board, move))
        table[str(board)] = curr_val, val_achieved, move
    return table

# helper function to build policy table 
# generate every board possible until the end of the game given a state 
def all_boards(state, results):
    possible_actions = actions(state)
    for action in possible_actions:
        results.append(result(state, action))
        if not is_terminal(result(state, action)):
            all_boards(result(state, action), results)
    return results


# PART 3: Design a graph data structure that lets you link nodes of neurons (units) in arbitrary ways as a directed graph.

# Layer Class 
# Creates a neural network layer for our graph
# each layer has an adjacency matrix, N x M matrix of weights 
#       N = number of nuerons on layer i 
#       M = number of nuerons on layer i-1 (the layer before, inputs)
# Will also have a vector for biases 
#   neurons = units in current layer 
#   inputs = units in previous layer 
class Layer:
    # initialize the weights and biases of the layer 
    def __init__(self, num_neurons, num_inputs):
        self.neurons = num_neurons

        np.random.seed(1000)

        # initialize weights to random value between -1 and 1 
        # shape of weight matrix is (inputs, neurons) – inputs rows, neurons columns
        #   creates a matrix of size N x M and fills it with random weights between -1 and 1
        #   this will be our adjaceny matrix
        #N = len(neurons)
        #M = len(inputs)
        self.weights = 2 * np.random.random((num_inputs,num_neurons)) - 1

        # initialize biases of the layer with random values between -1 and 1 in a separate vector 
        self.biases = 2 * np.random.random(num_neurons) - 1

# Directed_graph class --> builds our neural network 
# build graph as series of layers as defined above 
class Directed_Graph:

    # Two layered network
    def __init__(self, prev_layer, curr_layer):
        self.first_layer = prev_layer
        self.sec_layer = curr_layer

# PART 4
# Implement a classify function that takes an instance of the network and a vector of inputs, 
# and then ”runs” the network on that vector of inputs

# If you use adjacency matrices:
#   at each layer you’ll take the weight matrix and multiply it by the outputs from the previous layer 
#       NOTE: use inputs from the outside world, if it’s the first layer
# Add in the biases to get the values to input into your activation function

# Classify function 
#   takes in a network, a set of inputs, and an activation function 
# returns the outputs of that neuron (which would be inputs for the next layer)
def classify(network, input, activation):
    layer1_out = activation(np.dot(input, network.first_layer.weights) + network.first_layer.biases)
    layer2_out = activation(np.dot(layer1_out, network.sec_layer.weights) + network.sec_layer.biases)
    return layer1_out, layer2_out
    """
    # First find output of layer 1 
    out1 = np.dot(inputs, network.first_layer.weights) + network.first_layer.biases
    out1 = activation(out1)

    # feed that output into layer 2 
    out2 = np.dot(out1, network.sec_layer.weights) + network.sec_layer.biases
    out2 = activation(out2)

    #outputs holds the output for all layers in the network 
    return out1, out2
    """

# TWO ACTIVATION FUNCTIONS
# The sigmoid (or logistic or logit) function:
def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))
    """
    negated = [ -x for x in x]
    #print(negated)
    result = []
    for n in negated:
        result.append( 1 / (1 - exp(n)) )
    """
    #return result

# The ReLU (rectified linear unit) function:
#   ReLU outputs input if it is positive, otherwise outputs 0 
#   x is a matrix
def reLU(x):
    output = []
    for neuron in x:
        output.append(max(0,neuron))
    return output


# PART 5: update_weights 
# takes an instance of the network you designed in Part 3 and a vector of expected outputs
# uses back propagation to modify the weights in your network 
#   based on the differences between the expected outputs and the set of outputs obtained from the last call to classify.

def update_weights(network, out1, out2, expected, input, activation):
    
    # get the outputs from the network instance 
    #out1, out2 = classify(network, input, activation)

    # calculate the difference between our expected output and actual output 
    # aka calculate error 
    layer2_error = 2 * (expected - out2)

    # use this when we calculate delta 
    if activation == sigmoid: 
        delta2 = layer2_error * sig_deriv(out2)
    if activation == reLU:
        delta2 = layer2_error * relu_deriv(out2)
    
    # using a numpy to get dot product 
    # note that we started with layer 2 and fed that into layer 1 from back propagation 
    layer1_error = delta2.dot(network.sec_layer.weights.T)
    if activation == sigmoid: 
        delta1 = layer1_error * sig_deriv(out1)
    if activation == reLU:
        delta1 = layer1_error * relu_deriv(out1)

    # convert to np array for function use 
    delta1 = np.array(delta1)
    input = np.array([input])
    # calculate how much we need to adjust the weights by 
    add_weight1 = input.T.dot((delta1).reshape(1, network.first_layer.neurons))

    # repeat for layer 2
    delta2 = np.array(delta2)
    out1 = np.array([out1])
    add_weight2 = out1.T.dot((delta2).reshape(1, network.sec_layer.neurons))

    # adjust layers and biases
    network.first_layer.weights += add_weight1
    network.first_layer.biases += np.sum(delta1, axis=0)

    network.sec_layer.weights += add_weight2
    network.sec_layer.biases += np.sum(delta2, axis=0)

    return 0 

# ReLU derivative helper function as defined in the writeup 
def relu_deriv(x):
    # note that the input x is a list 
    out = []
    for val in x: 
        if val < 0:
            out.append(0)
        else:
            out.append(val)
    return out

# sigmoid derivative helper function 
def sig_deriv(x):
    sig_x = np.array(sigmoid(x))
    return sig_x * (1-sig_x)


# PART 6: Design a neural network that learns Hexapawn 
# 10 input nodes 
#   1 for specifying whose turn it is, 
#   9 for the state of the board
# 9 output nodes (for specifying which cells moving a pawn to would be optimal)
class Hex_Network():
    # create a network with 10 inputs, 9 outputs 
    # network has two layers 
    # sigmoid function based on performance does better than relu so set this as the default function 
    def __init__(self):
        self.first_layer = Layer(10,10)
        self.sec_layer = Layer(9,10)
        self.activation = sigmoid

    # Create a function that helps train the network we have created 
    def train(self, state):

        # given a current state, determine what the possible inputs are
        inputs = all_boards(state, [])

        # find out optimal outputs based on the policy table
        policy = build_policy(state)

        # determine a set of expected outputs for each of out boards
        expected = {}

        # our output only wants to be the cells not the turn so ommit the first value 
        for input in inputs:
            move = policy[str(input)][2]
            if move != None:
                out = result(input, move)
                expected[str(input)] = out[1:]
      
        trained = {}
        # train for 100 iterations 
        for input in inputs:
            if str(input) in expected:
                print("INPUT", input)
                print("EXPECTED", expected[str(input)])

                for i in range(10):
                    out1, out2 = classify(self, input, self.activation)
                    #update_weights(self, out1, out2, expected[str(input)], input, sigmoid)
                print(out1, out2)
                trained[str(input)] = self

        return trained

"""
*************** TRAINING HEXAPAWN ANALYSIS ********************
According to the textbook, all neural networks are sufficient with two layers
so for the case of Hexapawn we will use 2 layers as well.
Based on performance, the sigmoid function gives higher accuracy than reLU, so this is set as the 
default in the created Hex_Network class.

Given a state we can construct a policy statement that will provide us with the expected outputs 
for an optimal game. 
Using this we can train a network that can reasonably learn Hexapawn.

NOTE: when we run it with update_weights some of the cell numbers get really small (as they approach the 0)
Because of this sometimes numpy throws an error when raising exp() so for the time being the update_weights 
function call is commented out. 

With these weights, given a state, the network should be able to reasonably determine what the optimal move is
"""

def main():
    print("****************** TESTING PART 1 **********************")
    # NOTE: the state description is listed above in the coding section 

    initial_state1 = [1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    print("ORIGINAL BOARD")
    board = to_board(initial_state1)
    for row in board:
        print(row)

    # TESTING ACTIONS(s)
    action1 = actions(initial_state1)
    print("------ POSSIBLE ACTIONS -------")
    print(action1)

    # TESTING REUSLT(s, a)
    play1 = [2, 1, 'up']
    move = result(initial_state1, play1)
    #print(move)

    print("NOW POSSIBLE ACTIONS ARE")
    action2 = actions(move)
    print(action2)

    # IS_TERMINAL(s)
    terminal = [0, 1, 0,-1, 0, -1, 0, 0, 0]
    print("TERMINAL?")
    print(is_terminal(terminal))

    # UTILITY(s)
    print("utility of board where white won")
    print(utility(terminal))

    b_win = [1, 0, 0,-1, 0, 0, 0, -1, 0]
    print("utility of board where black won")
    print(utility(b_win))

    print("utility of an unfinished board:")
    print(utility(initial_state))
    
    print("***************** PART 2 ************************")
    print("INITIAL BOARD")
    board = to_board(initial_state1)
    for row in board:
        print(row)
    best_action = minimax_search(initial_state1)
    print('BEST ACTION')
    print(best_action)

    # see if search can determine the best step when one step away from winning 
    print("ALMOST WINNING BOARD")
    almost = [1, 0, -1, -1, -1, 1, 0, 1, 0, 1]
    print("possible actions", actions(almost))
    board = to_board(almost)
    for row in board:
        print(row)
    best_action = minimax_search(almost)
    print('BEST ACTION')
    print(best_action)
    
    print("----------- BUILDING OUR POLICY TABLE ------------")
    print("INITIAL STATE", initial_state)
    #all = all_boards(initial_state, [])

    policy = build_policy(initial_state) 
    for state in policy:
        print("state:", state, '\ncurr val, val achieved, best move = ', policy[state])
        print("--------------------------------------------------------------")
   
    print("*************** PART 4 *******************")  
    neurons = [1,2,3,4,5]
    inputs = [6,7,8,9,10]
    layer = Layer(5, 5)
    print("----- WEIGHTS ------")
    print(layer.weights)
    print("------- BIASES -------")
    print(layer.biases)

    #print(inputs*layer.weights)

    print("------ TESTING ACTIVATION FUNCTIONS --------")

    print("SIGMOID")
    s_test = [1,2,3,4,5]
    s_out = sigmoid(s_test)
    print(s_out)

    print("RELU")
    r_test = [1,2,3,4]
    r_out = reLU(r_test)
    print(r_out)

    # CREATING LAYERS 
    neurons1 = [1,2] 
    neurons2 = [3,4]

    input1 = [5,6]
    input2 = [7,8]

    layer1 = Layer(2, 2)
    #print("layer 1 weights: ", layer1.weights)
    layer2 = Layer(2, 2)
    #print("layer 2 weights: ", layer2.weights)

    # CREATING A DIRECTED GRAPH FROM THOSE LAYERS 
    network = Directed_Graph(layer1, layer2)

    # CLASSIFYING THE NETWORK WE CREATED 
    ins = [0,0]
    
    print("------------ CLASSIFY ------------")

    # Test classify function with a single iteration (this is going to be inaccurate)

    output1, output2 = classify(network, ins, sigmoid)   # TESTING WITH SIGMOID
    print("layer 1 output: \n", output1)
    print("layer 2 output: \n", output2)

    outputr1, outputr2 = classify(network, ins, reLU)  # TESTING WITH RELU 
    print("layer 1 output: \n", outputr1)
    print("layer 2 output: \n", outputr2)


    print("*********************** PART 5 ************************")
    # Testing an actual neural network and update weights function 

    # build a network 
    train_n = [0,0]
    train_i = [0,0,0]

    t_layer1 = Layer(2, 2)
    t_layer2 = Layer(2, 2)
    t_network = Directed_Graph(t_layer1, t_layer2)

    # TESTING USING 2 BIT ADDER EXAMPLE 
    print("---------- TRAINING -------------")
    examples = {'input': [[0,0], [0,1], [1,0], [1,1]], 'output': [[0,0], [0,1], [1,0], [1,1]]}
    
    print("ADDER INPUTS: ")
    print(examples['output'])

    print("START TRAINING")
    # train for 100 iterations and see if we can get close to desired output 
    for i in range(100):
        out1, out2 = classify(t_network, [1,1], sigmoid)
        update_weights(t_network, out1, out2, [1,1], [1,0], sigmoid)

    # check output
    # layer 2 output expected [1,1] (should be relatively close to this from training)
    print("layer1 output: \n", out1, "\nlayer 2 output: (aka our expected value) \n", out2)
    
    # Print out resulting weights and biases after training 
    result1 = t_network.first_layer
    result2 = t_network.sec_layer

    print("layer 1 weights: ")
    print(result1.weights) 
    print("layer 1 biases: ")
    print(result1.biases)

    print("layer 2 weights:")
    print(result2.weights)
    print("layer 2 biases:")
    print(result2.biases)


    print("**************** PART 6 *****************")

    print("------ different num inputs and neurons ------")
    l1 = Layer(10, 10)
    l2 = Layer(9, 10)
    net = Directed_Graph(l1, l2)

    input1 = [1,2,3,4,5,6,7,8,9,10]

    # check if classify works
    output1, output2 = classify(net, input1, sigmoid)
    print("LAYER 1 OUT ")
    print(output1)
    print("LAYER 2 OUT ")
    print(output2)

    # check if update weights works 
    update_weights(net, output1, output2, [1,2,3,4,5,6,7,8,9], input1, sigmoid)
    
    print("-------------------TRAINING HEXAPAWN-------------------------")
    hex_net = Hex_Network()
    hex_start_board = [1,-1,-1,-1,0,0,0,1,1,1]
    output1, output2 = classify(hex_net, hex_start_board, sigmoid)

    #hex_net.train(hex_start_board)

    # just a single example of one hexapawn input being trained against the expected optimal move as output
    # we can repeat this process with all boards to get a fully trained model 
    # a function is defined in Hex_Network class to do this for all boards
    pol = build_policy(hex_start_board)
    expected = result(hex_start_board, [0, 1, 'down'])
    expected = expected[1:]

    # train on this example 
    for i in range(100):
        out1, out2 = classify(hex_net, hex_start_board, sigmoid)
        update_weights(hex_net, out1, out2, expected, hex_start_board, sigmoid)
    
    # call classify on our trained network 
    out1, out2 = classify(hex_net, hex_start_board, sigmoid)
    result1 = hex_net.first_layer
    result2 = hex_net.sec_layer

    print("----------- LAYER 1 OUT ------------")
    print(output1)
    print("----------- LAYER 2 OUT ------------")
    print(output2)

    # print the trained weights and outputs
    """
    print("---------- layer 1 weights: ------------")
    print(result1.weights) 
    print("---------- layer 1 biases: ----------")
    print(result1.biases)

    print("----------- layer 2 weights: -----------")
    print(result2.weights)
    print("----------- layer 2 biases: -----------")
    print(result2.biases)
    """

    # based on the outputs we can see the most optimal cells that we tend to move to 

if __name__ == "__main__":
   main()