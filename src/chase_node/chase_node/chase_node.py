import os
import sys
import uuid
import json
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy as History
from rclpy.qos import QoSDurabilityPolicy as Durability
from rclpy.qos import QoSReliabilityPolicy as Reliability

from rmf_task_msgs.msg import ApiRequest, ApiResponse
from rmf_fleet_msgs.msg import FleetState

import random
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray

from shapely.geometry import Polygon, Point
from transitions import Machine

from chase_node.chase_strategy_core import ChaseStrategyCore

###############################################################################

class chase_node_class(Node):

    def __init__(self):
        super().__init__('chase_node')

        # Declare and get parameters
        self.declare_parameter('nav_graph_file_path', '/home/user/rmf_wakayama-u-farm-123/0.yaml')
        self.declare_parameter('boundary_map_file_path', '/home/user/rmf_wakayama-u-farm-123/1.yaml')
        self.declare_parameter('goal_threshold', 0.5)
        self.declare_parameter('target_threshold', 2.0)
        self.declare_parameter('target_invade_time_threshold', 3.0)
        #self.declare_parameter('use_sim_time', False)

        self.nav_graph_file_path = self.get_parameter('nav_graph_file_path').get_parameter_value().string_value
        self.boundary_map_file_path = self.get_parameter('boundary_map_file_path').get_parameter_value().string_value
        self.goal_threshold = self.get_parameter('goal_threshold').get_parameter_value().double_value
        self.target_threshold = self.get_parameter('target_threshold').get_parameter_value().double_value
        self.target_invade_time_threshold = self.get_parameter('target_invade_time_threshold').get_parameter_value().double_value
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value

        # Validate parameters
        if not os.path.exists(self.nav_graph_file_path):
            self.get_logger().error(f'Nav graph file not found: {self.nav_graph_file_path}')
            raise FileNotFoundError(self.nav_graph_file_path)
        if not os.path.exists(self.boundary_map_file_path):
            self.get_logger().error(f'Boundary map file not found: {self.boundary_map_file_path}')
            raise FileNotFoundError(self.boundary_map_file_path)
        
        self.get_logger().info(f'Using nav graph: {self.nav_graph_file_path}')
        self.get_logger().info(f'Using boundary map: {self.boundary_map_file_path}')
        self.get_logger().info(f'Goal threshold: {self.goal_threshold}m')
        self.get_logger().info(f'Target threshold: {self.target_threshold}m')
        self.get_logger().info(f'Target invade time threshold: {self.target_invade_time_threshold}s')

        self.transient_qos = QoSProfile(
            history=History.KEEP_LAST,
            depth=10,
            reliability=Reliability.RELIABLE,
            durability=Durability.TRANSIENT_LOCAL)
 
        self.fleet_states_subscription = self.create_subscription(
            FleetState,
            'fleet_states',
            self.fleet_state_callback,
            10)
        self.get_debug_target_subscription = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.get_debug_target_callback,
            10)
        self.get_mmw_target_subscription = self.create_subscription(
            MarkerArray,
            'tracker_marker_array',
            self.get_mmw_target_callback,
            10)
        self.task_res_subscription = self.create_subscription(
            ApiResponse, 
            'task_api_responses', 
            self.receive_response, 
            self.transient_qos)
        
        self.task_publisher = self.create_publisher(ApiRequest, 'task_api_requests', self.transient_qos)
        self.msg_id = uuid.uuid4()
        
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.free_robot_list = []
        self.ongoing_task_dic = {} #{'task_id':{'robot_name','goal_place'},'task_id':{'robot_name','goal_place'}...}
        self.cancelling_task_dic = {} #{'task_id':'cancel_request_id',task_id':'cancel_request_id',task_id':'cancel_request_id'...}

        with open(self.nav_graph_file_path, 'r') as file: #openrmf nav graph yaml file path
            nav_graph_data = yaml.safe_load(file)

        self.level_name = list(nav_graph_data['levels'].keys())[0]  
        self.vertices_dict = {vertex[2]['name']: [vertex[0], vertex[1]] for vertex in nav_graph_data['levels'][self.level_name]['vertices']}
        self.target_position = [None, None]

        with open(self.boundary_map_file_path, 'r') as file: #rmf nav graph yaml file describing the chase zone boundary
            boundary_map_data = yaml.safe_load(file)
        boundary_level_name = list(boundary_map_data['levels'].keys())[0]
        boundary_vertices = boundary_map_data['levels'][boundary_level_name]['vertices']
        self.chase_zone = Polygon([(vertex[0], vertex[1]) for vertex in boundary_vertices])

        self.strategy_core = ChaseStrategyCore(
            logger = self.get_logger,
            vertices_dict = self.vertices_dict,
            chase_zone = self.chase_zone,
            target_threshold = self.target_threshold,
            target_invade_time_threshold = self.target_invade_time_threshold,
            free_robot_list = self.free_robot_list,
            ongoing_task_dic = self.ongoing_task_dic,
            cancelling_task_dic = self.cancelling_task_dic,
            target_position = self.target_position,
            chase_task_requester = self.chase_task_requester,
            cancel_task_requester = self.cancel_task_requester
        )

        self.first_task_sent = False
#-----------------------------------------------------------------------------------#
    def fleet_state_callback(self, msg):
        # troble: sometime the fleet state msg will miss some robots' info, which 
        # will cause the node to think the robot is free and assign task to it, 
        # but in reality it's still on going task, which will cause the task can never 
        # be completed and never be removed from ongoing task list, and the robot 
        # will never be free again. A possible solution is to check the robot's position 
        # and compare with the goal position of its assigned task, if the distance is 
        # less than a threshold, consider the task is complete and remove it from ongoing 
        # task list, otherwise consider it's still on going task and should not be added to 
        # free robot list. This solution is implemented in the following code.
        for item in msg.robots:
            #self.get_logger().info(f"get robot msg: {item.name}")
            exists = any(item.name == obj.name for obj in self.free_robot_list)
            if len(self.ongoing_task_dic) != 0: 
                robot_position = np.array([item.location.x,item.location.y])
                
                current_task_id = next((task_id for task_id, task_info in self.ongoing_task_dic.items() 
                        if task_info['robot_name'] == item.name), None)

                if current_task_id == None:
                    return

                goal_position = np.array(self.vertices_dict[self.ongoing_task_dic[current_task_id]['goal_place']])
                distance = np.linalg.norm(robot_position - goal_position)

                if distance < self.goal_threshold: # consider as task complete when close enough to goal, remove from ongoing task list
                    if not exists:
                        self.free_robot_list.append(item)
                    task_id_to_delete = [task_id for task_id, task_info in self.ongoing_task_dic.items() if task_info['robot_name'] == item.name]
                    del self.ongoing_task_dic[task_id_to_delete[0]]
                    self.get_logger().info(f"task {task_id_to_delete[0]} complete, robot {item.name} is now free\n")
                else: # still on going task, should be removed in free robot list
                    if exists:
                        self.free_robot_list = list(filter(lambda obj: obj.name != item.name, self.free_robot_list))   
                        self.get_logger().info(f"robot {item.name} is on going task {current_task_id}\n")
            else: # no ongoing task, any robot should be in free robot list
                if not exists: # new robot available, add to free robot list
                    self.free_robot_list.append(item)
#-----------------------------------------------------------------------------------#
    def get_debug_target_callback(self, msg):
        self.target_position[0] = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info('target: ' + str(self.target_position))
#-----------------------------------------------------------------------------------#
    def get_mmw_target_callback(self, msg):
        marker_positions = []
        for marker in msg.markers:
            marker_positions.append([marker.pose.position.x, marker.pose.position.y])
        self.target_position[0] = marker_positions[0]
        self.get_logger().info('get mmw msg: ' + str(marker_positions[0]))
#-----------------------------------------------------------------------------------#
    def timer_callback(self):
        # self.get_logger().info('fsm state: ' + str(self.state)) # print status
        self.get_logger().info('free list: ') # print status
        for item in self.free_robot_list:
            self.get_logger().info(item.name)
        self.get_logger().info(' ')
        self.get_logger().info('on going task: ')
        if len(self.ongoing_task_dic) != 0:
            for task, task_info in self.ongoing_task_dic.items():
                self.get_logger().info(f"{task}, goal: {task_info['goal_place']}, robot: {task_info['robot_name']}")
        self.get_logger().info(' ')
        self.get_logger().info('on cancelling task: ')
        if len(self.cancelling_task_dic) != 0:
            for task, req_id in self.cancelling_task_dic.items():
                self.get_logger().info(f"{task}")
        self.get_logger().info('----------------------------------')

        self.strategy_core.strategy_update()
#-----------------------------------------------------------------------------------#
    def chase_task_requester(self, robot, fleet, starttime, places, rounds):
        # enable ros sim time
        if self.use_sim_time:
            self.get_logger().info("Using Sim Time")
            param = Parameter("use_sim_time", Parameter.Type.BOOL, True)
            self.set_parameters([param])

        # Construct task
        msg = ApiRequest()
        msg.request_id = "patrol_chase_" + str(uuid.uuid4())
        payload = {}

        if robot and fleet:
            self.get_logger().info("Using 'robot_task_request'")
            payload["type"] = "robot_task_request"
            payload["robot"] = robot
            payload["fleet"] = fleet
        else:
            self.get_logger().info("Using 'dispatch_task_request'")
            payload["type"] = "dispatch_task_request"
        
        request = {}

        # Set task request start time
        now = self.get_clock().now().to_msg()
        now.sec = now.sec + starttime
        start_time = now.sec * 1000 + round(now.nanosec/10**6)
        request["unix_millis_earliest_start_time"] = start_time
        # todo(YV): Fill priority after schema is added

        # Define task request category
        request["category"] = "patrol"

        # Define task request description
        description = {
            'places': places,
            'rounds': rounds
        }
        request["description"] = description
        payload["request"] = request
        msg.json_msg = json.dumps(payload)

        self.transient_qos.depth = 10

        print(f"Json msg payload: \n{json.dumps(payload, indent=2)}")
        self.task_publisher.publish(msg)

        self.first_task_sent = True

        #self.pending_task_list.append(msg)
#-----------------------------------------------------------------------------------#
    def cancel_task_requester(self, task_id):
        # Construct task
        msg = ApiRequest()
        msg.request_id = "cancel_task_" + str(uuid.uuid4())
        payload = {}
        payload['type'] = 'cancel_task_request'
        payload['task_id'] = task_id

        msg.json_msg = json.dumps(payload)
        print(f"Json msg payload: \n{json.dumps(payload, indent=2)}")
        self.task_publisher.publish(msg)

        self.cancelling_task_dic[task_id] = str(msg.request_id)
#-----------------------------------------------------------------------------------#
    def receive_response(self, response_msg: ApiResponse):
        print(f'Got response:\n{response_msg}')
        if self.first_task_sent: #to prevent unwanted msg
            if "patrol_chase" in response_msg.request_id:
                self.ongoing_task_dic[response_msg.request_id] = {
                    'robot_name' : json.loads(response_msg.json_msg).get('state').get('assigned_to').get('name'),
                    'goal_place' : json.loads(response_msg.json_msg).get('state').get('detail').get('places')[-1]
                }
                print("assign task complete\n")
            if "cancel_task" in response_msg.request_id:
                task_id_to_delete = [task_id for task_id, cancel_request_id in self.cancelling_task_dic.items() if cancel_request_id == response_msg.request_id]
                del self.ongoing_task_dic[task_id_to_delete[0]]
                del self.cancelling_task_dic[task_id_to_delete[0]]
                print("cancel task complete\n")
 #-----------------------------------------------------------------------------------#       
def main(argv=sys.argv):
    rclpy.init(args=sys.argv)

    chase_node = chase_node_class()
    rclpy.spin(chase_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)