
import heapq

# Define the Node class
class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.priority = cost + heuristic

    def __lt__(self, other):
        return self.priority < other.priority


def get_neighbors(node):

    graph = {
        'S': ['A', 'C', 'B'],
        'A': ['D', 'H' ],
        'B': ['C', 'E', 'F'],
        'C': ['B', 'E'],
        'D': ['A','E'],
        'E': ['D', 'B'],
        'F': ['I', 'K', 'M'],
        'I': ['L', 'F'],
        'K': ['F', 'M', 'L'],
        'M': ['F', 'K', 'O'],
        'O': ['L', 'G'],
        'L': ['I', 'K', 'O','G'],
        'H': ['A', 'J','N'],
        'J': ['I', 'H','N'],
        'N': ['H', 'J','G'],
        'G': [None]
        
    }
    
    return graph[node[0]]



def get_coordinates(node):
    pass
    
    

#%% A Star
# Define the A* search function
def a_star_search(start_state, goal_state, get_neighbors, cost_func, heuristic_func):
    open_list = []
    closed_set = set()

    start_node = Node(start_state, cost=0, heuristic=heuristic_func(start_state))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]  # Reverse the path to get the correct order

        closed_set.add(current_node.state)

        neighbors = get_neighbors(current_node.state)
        for neighbor_state in neighbors:
            if neighbor_state in closed_set:
                continue

            cost_to_neighbor = current_node.cost + cost_func(current_node.state, neighbor_state)
            neighbor_node = Node(
                state=neighbor_state,
                parent=current_node,
                cost=cost_to_neighbor,
                heuristic=heuristic_func(neighbor_state),
            )

            heapq.heappush(open_list, neighbor_node)

    return None  # No path found


def cost_func(state1, state2):
    # Define a constant cost for moving from state1 to state2string_to_number.get(string, -1)

    costmap = {
        "apple": 1,
        "banana": 2,
        "cherry": 3,
        "date": 4,
    }
    # Add more mappings as needed
    return costmap.get(state1+state2)

def heuristic_func(state):
    # Use Manhattan distance as a heuristic (estimated cost to reach the goal)
    x, y = state
    goal_state = (0, 13)  # Define the goal state
    return abs(x - goal_state[0]) + abs(y - goal_state[1])

def main():
    # Sample start and goal states
    start_state = (0, 5)
    goal_state = (0, 13)

    # Find the path using A* search
    path = a_star_search(start_state, goal_state, get_neighbors, cost_func, heuristic_func)
    if path:
        print("Path:", path)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()

