import sys
import uuid
import argparse
import json
import asyncio
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_system_default
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy as History
from rclpy.qos import QoSDurabilityPolicy as Durability
from rclpy.qos import QoSReliabilityPolicy as Reliability

from rmf_task_msgs.msg import ApiRequest, ApiResponse
from rmf_fleet_msgs.msg import RobotState
from rmf_fleet_msgs.msg import FleetState

import random
from geometry_msgs.msg import PoseStamped
###############################################################################

class chase_node_class(Node):

    def __init__(self):
        super().__init__('chase_node')

        self.use_sim_time = True

        self.response = asyncio.Future()

        self.transient_qos = QoSProfile(
            history=History.KEEP_LAST,
            depth=1,
            reliability=Reliability.RELIABLE,
            durability=Durability.TRANSIENT_LOCAL)

        self.fleet_states_subscription = self.create_subscription(
            FleetState,
            'fleet_states',
            self.fleet_state_callback,
            10)
        self.get_target_subscription = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.get_target_callback,
            10)
        self.task_res_subscription = self.create_subscription(
            ApiResponse, 
            'task_api_responses', 
            self.receive_response, 
            self.transient_qos)
        
        self.task_publisher = self.create_publisher(ApiRequest, 'task_api_requests', self.transient_qos)
        
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.free_robot_list = []

        with open('/home/a/rmf_maps/simulation/nav_graph/0.yaml', 'r') as file:
            nav_graph_data = yaml.safe_load(file)

        self.vertices_dict = {vertex[2]['name']: [vertex[0], vertex[1]] for vertex in nav_graph_data['levels']['iwadefarm']['vertices']}
        self.vertices_list = list(self.vertices_dict.items())

        self.target_position = None

    def fleet_state_callback(self, msg):
        for item in msg.robots:
            exists = any(item.name == obj.name for obj in self.free_robot_list)
            if item.task_id == '':
                if not exists: 
                    self.free_robot_list.append(item)
            else:
                if exists:
                    self.free_robot_list = list(filter(lambda obj: obj.name != item.name, self.free_robot_list))

    def get_target_callback(self, msg):
        self.target_position = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info('target: ' + str(self.target_position))

    def timer_callback(self):
        self.get_logger().info('free list: ')
        for item in self.free_robot_list:
            self.get_logger().info(item.name)
        self.get_logger().info(' ')

        if self.target_position != None:
            self.get_logger().info('get it!')

            names = list(self.vertices_dict.keys())
            coordinates = np.array(list(self.vertices_dict.values()))
            squared_distances = np.sum((coordinates - np.array(self.target_position)) ** 2, axis=1)
            min_index = np.argmin(squared_distances)
            sorted_indices = np.argsort(squared_distances)

            i = 0
            for item in self.free_robot_list:
                self.task_requester(
                    fleet = 'tinyRobot',
                    robot = item.name,
                    places = [names[sorted_indices[i]]],
                    rounds = 1,
                    starttime = 0)
                i = i + 1 
            
            self.target_position = None

        # for item in self.free_robot_list:
        #     self.task_requester(
        #         fleet = 'tinyRobot',
        #         robot = item.name,
        #         places = [random.choice(self.vertices_list)[0]],
        #         rounds = 1,
        #         starttime = 0)



    def task_requester(self, robot, fleet, starttime, places, rounds):
        # enable ros sim time
        if self.use_sim_time:
            self.get_logger().info("Using Sim Time")
            param = Parameter("use_sim_time", Parameter.Type.BOOL, True)
            self.set_parameters([param])

        # Construct task
        msg = ApiRequest()
        msg.request_id = "patrol_" + str(uuid.uuid4())
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
    
    def receive_response(self, response_msg: ApiResponse):
            if response_msg.request_id == "patrol_" + str(uuid.uuid4()):
                self.response.set_result(json.loads(response_msg.json_msg))

def main(argv=sys.argv):
    rclpy.init(args=sys.argv)

    chase_node = chase_node_class()
    rclpy.spin(chase_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
