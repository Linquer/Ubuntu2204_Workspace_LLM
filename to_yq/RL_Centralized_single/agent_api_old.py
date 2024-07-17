import json
import logging
import re

import torch
# from openai import OpenAI

# from RL_Centralized_single.constant import Prompt


class Agent_Api:
    def __init__(self):
        self.SIGN = 2
        # self.client = OpenAI(
        #     # This is the default and can be omitted
        #     base_url="https://api.gpt.ge/v1/",

        # )
        # environment_describe = Prompt.environment_describe
        # self.messages = [
        #     {"role": "system", "content": f"{environment_describe}"},
        # ]

    # def parse_json(self, input_str: str):
    #     pattern = r'```json(.*?)```'
    #     match = re.search(pattern, input_str, re.DOTALL)
    #     if match:
    #         json_str = match.group(1).strip()
    #         try:
    #             return json.loads(json_str)
    #         except json.JSONDecodeError:
    #             return None
    #     else:
    #         return None
    def get_action(self, state):
        if self.SIGN == 2:
            self.SIGN = 1
            return 2
        elif self.SIGN == 1:
            self.SIGN = 0
            return 1
        else:
            self.SIGN = 2
            return 0
        
    # def get_action(self, state):
    #     n = len(state) / 2
    #     self.messages.append({"role": "user", "content": f"flow num:{n}, state:{state}"})
    #     response = self.client.chat.completions.create(
    #         model="gpt-4o",
    #         response_format={"type": "json_object"},
    #         messages=self.messages
    #     )
    #     action = response.choices[0].message.content
    #     logging.info(f'action: {action}')
    #     json_action = json.loads(action)
    #     action = json_action['action']
    #     logging.info(f'action: {action}')
    #     # action = self.parse_json(action)['action']
    #     self.messages.append({"role": "assistant", "content": f"{action}"})
    #     if 0 <= int(action) <= n:
    #         return int(action)
    #     else:
    #         exit('wrong action!')
