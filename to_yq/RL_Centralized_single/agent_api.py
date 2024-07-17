import json
import logging
import re
import numpy as np
import torch

class Agent_Api:
    def __init__(self):
        self.alpha = 1.0
        self.beta = 0.6
        self.gamma = 0.2
        self.delta = 0.2
        self.prob = 0
        self.flag_ac = False
        self.action_count = []

    ## 参数
    def get_action(self, state):
        n = int(len(state) / 4)
        w_arr = np.zeros(n)
        for i in range(n):
            w_arr[i] = self.alpha * state[i] + self.beta * state[n + i] + \
                        self.gamma * state[2 * n + i] + self.delta * state[3 * n + i]   
        action = self.random_argmax(w_arr, state)
        if action == 0:
            return 0
        e_greedy_action = self.e_greedy(n, self.prob)
        if e_greedy_action:
            if state[e_greedy_action] != 0:
                action = e_greedy_action
        return action
    
    
    def e_greedy(self, n, prob=0):
        if np.random.rand() < prob:
            return np.random.randint(1, n + 1)
        else:
            return 0
    # 实现argmax不再选择第一个最大值的index
    def random_argmax(self, arr, state):
        max_value = np.max(arr)
        indices = np.where(arr == max_value)[0]
        # 即便是权重最大的，如果urgency为0也不选择该动作
        need_delete_index = []
        for i in range(len(indices)):
            if state[indices[i]] == 0:
                need_delete_index.append(i)
        indices = np.delete(indices, need_delete_index)
        if len(indices) == 0:
            return 0
        random_index = np.random.randint(len(indices))
        return indices[random_index] + 1

    ## 代码
    # def get_action(self, state):
    #     Num = int(len(state) / 2) # calculate the number of data flows
    #     ### Begin ###
    #     # Your code here
    #     # return Int # Output as an integer between [1- N], representing the number of the called data flow.
    #     ### End ###

    # 实验代码
    # def get_action(self, state):
    #     N = int(len(state) / 2)  # calculate the number of data flows
    #     ### Begin ###
    #     if self.flag_ac == False:
    #         self.action_count = [0] * N
    #         self.flag_ac = True
    #     # Separate the urgency and packet arrival factor
    #     urgency = state[:N]
    #     arrival_flag = state[N:]

    #     # Initialize a list to hold the priority scores
    #     priority_scores = []

    #     # Calculate a priority score for each data flow
    #     for i in range(N):
    #         # Include arrival_flag in the priority calculation to prefer data flows with new packets
    #         priority = urgency[i] + arrival_flag[i] * 10  # give higher weight to newly arrived packets
    #         priority_scores.append((priority, i + 1))
        
    #     # Sort by priority (highest first), then by data flow index to ensure fairness
    #     priority_scores.sort(reverse=True, key=lambda x: (x[0], -x[1]))

    #     # Select the data flow with the highest priority score
    #     selected_flow = priority_scores[0][1]

    #     # Ensure no flow is chosen too frequently
    #     if self.action_count[selected_flow - 1] > sum(self.action_count) / len(self.action_count):
    #         for score, flow in priority_scores:
    #             if self.action_count[flow - 1] <= sum(self.action_count) / len(self.action_count):
    #                 selected_flow = flow
    #                 break
        
    #     self.action_count[selected_flow - 1] += 1
    #     return selected_flow # Output as an integer between [1- N], representing the number of the called data flow.
    #     ### End ###





if __name__ == '__main__':
    state = [0, 2, 1, 1]
    agent = Agent_Api()
    action = agent.get_action(state)
    print(action)