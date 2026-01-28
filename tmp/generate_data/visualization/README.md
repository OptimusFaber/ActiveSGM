rgbd: GT RGB-D visualization
render_rgbd: render RGB-D visualization
poses: (np.ndarray, [4,4]). Camera-to-world. RUB system. 
planning_path: (np.ndarray, [N,4,4]), each element is a planning pose 
lookat_tgts: (np.ndarray, [N,3]), uncertaint target observation locations to lookat.  
state: (str), planner state 
