import numpy as np
from shapely.geometry import Polygon, Point
from transitions import Machine

class ChaseStrategyCore:
    def __init__(
        self, logger,
        vertices_dict, chase_zone, 
        target_threshold, target_invade_time_threshold, 
        free_robot_list, ongoing_task_dic, cancelling_task_dic, target_position, 
        chase_task_requester, cancel_task_requester
    ):    
        self.free_robot_list = []
        self.ongoing_task_dic = {} #{'task_id':{'robot_name','goal_place'},'task_id':{'robot_name','goal_place'}...}
        self.cancelling_task_dic = {} #{'task_id':'cancel_request_id',task_id':'cancel_request_id',task_id':'cancel_request_id'...}
        self.target_position = [None, None]
        self.get_logger = logger
        self.request_task_handler = chase_task_requester
        self.cancel_task_handler = cancel_task_requester

        self.free_robot_list = free_robot_list      
        self.ongoing_task_dic = ongoing_task_dic    
        self.cancelling_task_dic = cancelling_task_dic
        self.target_position = target_position

        self.chase_zone = chase_zone
        self.vertices_dict = vertices_dict
        self.target_threshold = target_threshold
        self.target_invade_time_threshold = target_invade_time_threshold

        self.target_invade_time = 0
        states = ['standby','ambush', 'chasing', 'cancelling']
        transitions = [
            {'trigger': 'start_ambush',    'source': 'standby',    'dest': 'ambush',    'conditions': 'standby_to_ambush_conditions'},
            {'trigger': 'back_to_standby', 'source': 'ambush',     'dest': 'standby',   'conditions': 'ambush_to_standby_conditions'},
            {'trigger': 'start_new_chase', 'source': 'ambush',    'dest': 'chasing',   'conditions': 'ambush_to_chasing_conditions'},
            {'trigger': 'cancel_chase',    'source': 'chasing',    'dest': 'cancelling','conditions': 'chasing_to_cancelling_conditions'},
            {'trigger': 'resume_chase',    'source': 'cancelling', 'dest': 'chasing',   'conditions': 'cancelling_to_chasing_conditions'},
            {'trigger': 'stop_chase',      'source': 'cancelling', 'dest': 'standby',   'conditions': 'cancelling_to_standby_conditions'},
        ]
        self.machine = Machine(
            model=self, 
            states=states, 
            transitions=transitions, 
            initial='standby',
            ignore_invalid_triggers=True
        )
#-----------------------------------------------------------------------------------#
    def standby_to_ambush_conditions(self):
        # if len(self.cancelling_task_dic) == 0:
        if self.target_position[0] is not None:
            self.get_logger().info('get target position: ' + str(self.target_position[0]))
            if self.chase_zone.contains(Point(self.target_position[0])):
                self.get_logger().info('a target just invade chase zone, start ambush timer')
                self.target_position[1] = self.target_position[0]
                self.target_position[0] = None
                return True    
        self.target_position[0] = None

        return False

    def ambush_to_standby_conditions(self):
        self.get_logger().info('ambush timer: ' + str(self.target_invade_time))
        if self.target_position[0] is not None:
            if not self.chase_zone.contains(Point(self.target_position[0])):
                self.get_logger().warn('target out of chase zone, back to standby')
                self.target_position[1] = None
                self.target_position[0] = None
                self.target_invade_time = 0
                return True

        if self.target_invade_time >= self.target_invade_time_threshold:
            self.get_logger().warn('target invade time exceed threshold, back to standby')
            self.target_position[1] = None
            self.target_position[0] = None
            self.target_invade_time = 0
            return True

        self.target_invade_time += 1
        return False
    
    def ambush_to_chasing_conditions(self):
        if self.target_position[0] is not None:
            if self.chase_zone.contains(Point(self.target_position[0])):
                self.get_logger().warn('target still in chase zone, start chasing')
                self.target_position[1] = self.target_position[0]
                self.target_position[0] = None
                return True

        return False

    def chasing_to_cancelling_conditions(self):
        if self.target_position[0] is not None:
            moved_distance = np.linalg.norm(np.array(self.target_position[0]) - np.array(self.target_position[1]))
            if moved_distance >= self.target_threshold:
                self.get_logger().info('target moved, distance: ' + str(moved_distance))
                #self.target_position[1] = self.target_position[0]
                #self.target_position[0] = None
                return True

        return False
            

    def cancelling_to_chasing_conditions(self):
        if len(self.cancelling_task_dic) == 0:
            if self.target_position[0] is not None:
                if self.chase_zone.contains(Point(self.target_position[0])):
                    self.get_logger().warn('target still in chase zone')
                    self.target_position[1] = self.target_position[0]
                    self.target_position[0] = None
                    return True

        return False

    def cancelling_to_standby_conditions(self):
        if self.target_position[0] is not None:
            if not self.chase_zone.contains(Point(self.target_position[0])):
                self.get_logger().warn('target out of chase zone')
                self.target_position[1] = None
                self.target_position[0] = None
                return True

        return False

#-----------------------------------------------------------------------------------#
    def on_enter_ambush(self):
        self.get_logger().info('----Entering ambush state----')
        self.get_logger().info('----------------------------------')

    def on_enter_chasing(self):
        self.get_logger().info('----Entering chasing state----')
        self.get_logger().info('----------------------------------')

        self.get_logger().info('get it!')

        names = list(self.vertices_dict.keys()) #find closest two waypoints to target
        coordinates = np.array(list(self.vertices_dict.values()))
        squared_distances = np.sum((coordinates - np.array(self.target_position[1])) ** 2, axis=1)
        # min_index = np.argmin(squared_distances)
        sorted_indices = np.argsort(squared_distances)

        closest_robot = [None,None] #find closest robots to the waypoints
        for i in range(0,2): #only closest two waypoints
            goal_position = np.array(list(self.vertices_dict[names[sorted_indices[i]]]))

            min_distance = float('inf')
            for item in self.free_robot_list:
                if item.name in closest_robot :
                    continue
                robot_position = np.array([item.location.x,item.location.y])
                distance = np.linalg.norm(robot_position - goal_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_robot[i] = item.name
            if closest_robot[i]:
                self.request_task_handler(
                    fleet = 'agilex_fleet',
                    robot = closest_robot[i],
                    places = [names[sorted_indices[i]]],
                    rounds = 1,
                    starttime = 0)            
    
    def on_enter_cancelling(self):
        self.get_logger().info('----Entering cancelling state----')
        self.get_logger().info('----------------------------------')

        if len(self.cancelling_task_dic) != len(self.ongoing_task_dic):
            self.get_logger().info("cancel on going task\n")
            for task_id, robot_name in self.ongoing_task_dic.items():
                self.cancel_task_handler(task_id)
    
    def on_enter_standby(self):
        self.get_logger().info('----Entering standby state----')
        self.get_logger().info('----------------------------------')

    def strategy_update(self):
        self.get_logger().info('fsm state: ' + str(self.state)) # print status
        self.start_ambush()         # conditions: a target just invade chase zone
        self.back_to_standby()      # conditions: target out of chase zone, target invade time exceed threshold
        self.start_new_chase()      # conditions: is_target_available
        self.cancel_chase()         # conditions: is_target_moved_threshold
        self.resume_chase()         # conditions: is_target_moved_away
        self.stop_chase()           # conditions: is_target_unavailable
