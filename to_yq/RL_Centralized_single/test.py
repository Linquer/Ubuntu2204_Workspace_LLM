import time
from datetime import datetime

from agent_api import Agent_Api
from config import CentralizedConfig
from environment import Environment
import sys
import logging
import numpy as np
import os
import argparse


def config_logging(file_name: str, console_level: int = logging.INFO, file_level: int = logging.INFO):
    file_handler = logging.FileHandler(file_name, mode='w', encoding="utf8")
    file_handler.setFormatter(logging.Formatter(
        '%(levelname)s %(asctime)s_%(filename)s[line:%(lineno)d] %(message)s'
    ))
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s %(asctime)s_%(filename)s[line:%(lineno)d] %(message)s',
    ))
    console_handler.setLevel(console_level)

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler],
    )


def parse_args():
    # parser = argparse.ArgumentParser(description='Run parameters')
    # parser.add_argument('--flow_num', type=int, default=8)
    # parser.add_argument('--traffic_id_begin', type=int, default=0)
    # parser.add_argument('--traffic_id_end', type=int, default=0)
    # parser.add_argument('--model_suffix', type=str, default="test")
    # parser.add_argument('--traffic_prefix', type=str, default="c_w_")
    # parser.add_argument('--traffic_exp_name', type=str, default="flow_8_flow_compare")

    parser = argparse.ArgumentParser(description='Run parameters')
    parser.add_argument('--flow_num', type=int, default=8)
    parser.add_argument('--traffic_id_begin', type=int, default=0)
    parser.add_argument('--traffic_id_end', type=int, default=99)
    parser.add_argument('--model_suffix', type=str, default="test")
    parser.add_argument('--traffic_prefix', type=str, default="c_")
    parser.add_argument('--traffic_exp_name', type=str, default="flow_8")

    return parser.parse_args()


# main：主程序
if __name__ == '__main__':
    args = parse_args()
    print(f"args : {args}")

    flow_num = args.flow_num
    traffic_id_begin = args.traffic_id_begin
    traffic_id_end = args.traffic_id_end
    model_suffix = args.model_suffix
    traffic_exp_name = args.traffic_exp_name
    traffic_prefix = args.traffic_prefix

    config_logging(file_name=f'test{datetime.now()}.log')

    logging.info(f"pid : {os.getpid()}")

    average_timely_throughput_list = np.array([])
    average_theory_throughput_list = np.array([])
    state_list = []

    for traffic_id in range(traffic_id_begin, traffic_id_end + 1):

        logging.info(f"traffic id : {traffic_id}")
        config = CentralizedConfig(flow_num=flow_num, traffic_id=traffic_id,
                                   model_path="./c_model",
                                   traffic_exp_name=traffic_exp_name,
                                   traffic_prefix=traffic_prefix)
        # 环境
        environment = Environment(config=config)
        # 智能体
        agent = Agent_Api()
        # 智能体与仿真时间初始化
        config.initialize_simulation_time(traffic_id=0)
        # stats info
        train_begin_time = 0
        train_end_time = 0
        total_train_time = 0
        # best info
        best_train_time = 0
        best_train_episode = 0
        best_timely_throughput = 0
        # decision time
        decision_times = []

        environment.initialize(config=config, traffic_id=0)
        present_state = environment.get_state(time_slot=1)
        action_cnt = [0 for _ in range(config.ACTION_DIM)]



        for time_slot in range(1, 120+1):
            state_list.append(present_state)

            # make action
            logging.info(f"state :{present_state}")
            time_before_decision = datetime.now()
            action = agent.get_action(state=present_state)
            time_after_decision = datetime.now()
            decision_times.append((time_after_decision - time_before_decision).total_seconds())
            logging.info(f'average_decision_time:{np.mean(decision_times)}')
            action_cnt[action] += 1
            # environment step and give feedback
            plot_state = present_state
            next_state, reward, end, expire_list, generate_list = environment.step(time_slot=time_slot, action=action)
            # agent.perceive(state=present_state, action=action, reward=reward)
            # update the present observation
            present_state = next_state

        total_send_before_expiration, \
            total_generate_packet_count, \
            mac_delay_from_queue_to_send, \
            mac_delay_from_head_to_send, \
            mac_delay_from_queue_to_send_success, \
            mac_delay_from_head_to_send_success = environment.get_stats()
        present_timely_throughput = total_send_before_expiration / total_generate_packet_count

        # print('(', episode, ') timely throughput : ', present_timely_throughput,
        #       'train time : ', total_train_time)
        # _success后缀的是指统计成功到达目的节点的数据包，最后用的_success
        logging.info('timely throughput : ' + str(present_timely_throughput) +
                     ',mac delay(queue, total packet) : ' + str(round(mac_delay_from_queue_to_send, 5)) +
                     'mac delay(head, total packet) : ' + str(round(mac_delay_from_head_to_send, 5)) +
                     'mac delay(queue, success packet) : ' + str(round(mac_delay_from_queue_to_send_success, 5)) +
                     'mac delay(head, success packet) : ' + str(round(mac_delay_from_head_to_send_success, 5)) +
                     'action cnt : ' + str(action_cnt))
        logging.info("End")
        average_timely_throughput_list = np.append(average_timely_throughput_list, present_timely_throughput)
        average_theory_throughput_list = np.append(average_theory_throughput_list, config.THEORY_TIMELY_THROUGHPUT)
    logging.info(np.array(state_list).mean(axis=0))
    logging.info(f"average_theory_throughput:{average_theory_throughput_list.mean()} ,average_timely_throughput:{average_timely_throughput_list.mean()}")
    print("State len: ", len(state_list))
    np.save("state_list_flow8_8.npy", state_list)