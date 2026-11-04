
class State:
    def handle(self):
        pass

class ScanningState(State):
    def handle(self):
        print("Robot Arm: Scanning Objects")

class GraspingState(State):
    def handle(self):
        print("Robot Arm: Grasping Object")

class DockingState(State):
    def handle(self):
        print("Robot Arm: Returning to Dock")

class RobotArm:
    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def execute(self):
        self.state.handle()

robot = RobotArm()
states = {
    1: ScanningState,
    2: GraspingState,
    3: DockingState
}

print("1. Scanning  2. Grasping  3. Docking")
choice = int(input("Enter Choice: "))

if choice in states:
    robot.set_state(states[choice]())
    robot.execute()
else:
    print("Invalid Choice")
