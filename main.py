import numpy as np
import copy
import random
import json
import csv

nodeLength = 20

row1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

wall1 = [0] * nodeLength
wall2 = [0] * nodeLength
wall3 = [0,1,2,3,4,5,6,7,8,9]

layers = [row1, wall1, wall2, wall3]

forwardsGlues = [[], [], []]
for i in range(len(layers)-1) :
    for j in range(len(layers[i])):
        forwardGlue = {}
        for k in range(len(layers[i+1])):
            randomNumber = random.uniform(0, 1)
            forwardGlue[k] = randomNumber
        forwardsGlues[i].append(forwardGlue)

def run_train(image, glue, correctNum) :
    row1 = copy.deepcopy(image)
    forwardsGlues = copy.deepcopy(glue)
    for i in range(len(row1)) :
        row1[i] = int(row1[i])
        row1[i] /= 255

    wall1 = [0] * nodeLength
    wall2 = [0] * nodeLength
    wall3 = [0,1,2,3,4,5,6,7,8,9]

    layers = [row1, wall1, wall2, wall3]

    for i in range(len(layers)-1) :
        for j in range(len(layers[i])):
            for k in range(len(layers[i+1])):
                layers[i+1][k] += layers[i][j] * forwardsGlues[i][j][k]

    correct_last_layer = [0,0,0,0,0,0,0,0,0,0]
    correct_last_layer[int(correctNum)] = 1

    actual_layers = copy.deepcopy(layers)
    correct_layers = [row1, [0] * nodeLength, [0] * nodeLength, correct_last_layer]
    for i in range(len(actual_layers)-1, 0, -1): # layer right
        for j in range (len(actual_layers[i])): # right
            for k in range(len(actual_layers[i-1])): # left
                percentage_increase = correct_layers[i][j] - actual_layers[i][j]
                percentage_increase /= len(actual_layers[i-1])
                glue = forwardsGlues[i-1][k][j]
                difference_in_glue = 1 - glue
                difference_in_glue *= percentage_increase
                forwardsGlues[i-1][k][j] += difference_in_glue
                forwardsGlues[i-1][k][j] /= 2
                correct_layers[i-1][k] = actual_layers[i][j] / forwardsGlues[i-1][k][j]
    return forwardsGlues


with open('archive/mnist_train.csv', mode='r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    counter = -1
    stop_condition = 100
    for row in csv_reader:
        print(counter)
        counter += 1
        if counter == 0:
            continue
        number = row.pop(0)
        forwardsGlues = run_train(row, forwardsGlues, number)
        if counter == stop_condition :
            break


filename = 'glue.json'          #use the file extension .json
with open(filename, 'w') as file_object:  #open the file in write mode
 json.dump(forwardsGlues, file_object)

def run_test(image, glue) :
    row1 = copy.deepcopy(image)
    for i in range(len(row1)) :
        row1[i] = int(row1[i])
        row1[i] /= 255

    wall1 = [0] * nodeLength
    wall2 = [0] * nodeLength
    wall3 = [0,1,2,3,4,5,6,7,8,9]

    layers = [row1, wall1, wall2, wall3]

    for i in range(len(layers)-1) :
        for j in range(len(layers[i])):
            for k in range(len(layers[i+1])):
                layers[i+1][k] += layers[i][j] * glue[i][j][str(k)]
    
    return layers[3]



model = json.load(open('glue.json'))

with open('archive/mnist_train.csv', mode='r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    counter = -1
    stop_condition = 10
    for row in csv_reader:
        counter += 1
        if counter == 0:
            continue
        number = row.pop(0)
        answer = run_test(row, model)
        print(number)
        biggest = [0, -1]
        for i in range(len(answer)):
            if answer[i] > biggest[0]:
                biggest = [answer[i], i] 
        print(biggest[1], answer)
        if counter == stop_condition :
            break