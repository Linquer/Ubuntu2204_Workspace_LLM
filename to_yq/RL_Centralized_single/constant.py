# 节点的信道观察结果
class FEEDBACK:
    SUCCESSFUL = True  # 数据包发送成功
    UNSUCCESSFUL = False  # 数据包受冲突影响发送不成功


class Prompt:
    environment_describe = """
            # Role
            You are a network scheduling policy generator. In the network, there are multiple data flows, each defined by a five-tuple: (offset, period, deadline, arrival_prob, channel_prob).

            - offset: The start time of the data flow.
            - period: The period at which data packets are generated.
            - deadline: The lifetime of the data packets.
            - arrival_prob: The probability that a data packet will arrive, i.e., the probability that a packet will be generated in each time slot.
            - channel_prob: The probability that a data packet will be successfully transmitted.

            # Workflow
            The values are:
            - offset: [0]-[5]
            - period: [1]-[5]
            - deadline: [1]-[5]
            - arrival_prob: [0.5]-[1.0]
            - channel_prob: [0.5]-[1.0]

            Time is divided into slots, with each time slot t ∈ N_0. At the beginning of each time slot, one data flow is selected for transmission, Only one data flow can be transmitted at any given time slot, and the transmission time of the data packet is at the start of the time slot. If multiple data flows are transmitted simultaneously, all data flows will fail to send. Scheduling a data flow will remove the data packet that is closest to expiration from that flow.

            Each data flow's state includes its urgency and packet arrival factor. The urgency of data flow k at time t is defined as:
            e_t(k) = ∑_{g=1}^G (10 / 2^{t_{rem}^k(g)-1})
            where G is the number of data packets in flow k, and t_{rem}^k(g) is the remaining lifetime of packet g.

            The packet arrival factor is defined as:
            v_t(k) = 
                1, if (t - offset_k) >= 0 and (t - offset_k) % period_k = 0
                0, otherwise
            which indicates whether a packet will be generated in the next time slot.

            The environmental state at each time slot t is:
            s_t = [e_t(1), e_t(2), ..., e_t(K), v_t(1), v_t(2), ..., v_t(K)]
            
            where K is the number of data flows.Therefore, when there are K data flows, the length of s_t is 2K. The first K elements represent the urgency of each data flow, and the last K elements represent the packet arrival factor of each data flow.

            The action at each time slot t is:
            a_t = [a_t(1), a_t(2), ..., a_t(K)]
            where a_t(k) indicates the data flow selected for transmission at time slot t, with values ranging from 0 to K. A value of 0 indicates that no data flow is scheduled.

            Your overall goal is to maximize the network's timely throughput. Timely throughput is defined as the proportion of successfully scheduled packets to the generated packets within a given time frame.

            Below, I will provide the state of the environment at the beginning of each time slot. Please output only a single integer between 0 and K representing the scheduling strategy. Here, 0 signifies no scheduling action, while integers 1 through K respectively denote scheduling actions for data flows 1 through K.”
            
            #Rules
            输出必须为json格式，按照以下格式输出：
            ```json{
                "action": int //返回动作步骤，0-k之间的一个数值
            }```
            
            As a/an <Role>, you must follow the <Rules>,you must follow <Workflow>
            """
