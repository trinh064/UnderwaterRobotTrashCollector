# WaterRoomba
CSCI 4951 Senior Design Project: Underwater Robotic manipulator

Introduction:<br/>
Pollution is an ever-growing problem in today’s world that is not fully recognized as the catastrophe that it is. Many bodies of water, and especially the oceans, have all kinds of garbage strewn throughout that is detrimental to the aquatic life that inhabits the waters. This garbage can be found at seemingly any depth, from floating on the surface all the way down to the bottom. There are many trash collection methods that exist today, but many of them only target the pollutants that are on the surface of the water. As a result, they miss a large majority of the problem that plagues the water. <br/>
        	The team developed a solution that would allow the objects under the surface to be detected autonomously. A convolutional neural network (CNN) is run on a Jetson Nano development board and paired with an Intel Realsense RGB-D camera to detect objects in its field of view. If the probability of a successful grasp is high enough, the gripper closes once the object is inside it. Probabilities that are output from the CNN represent the probability of a successful grasp if the gripper were to move straight down, and therefore closer to the object. Data is then output from the Jetson board to various servo motors to control and close the gripper. All of this was done within a $1000 budget that was given to us by Professor Choi’s research funds, and the project came in under budget.<br/>

Instruction: <br/>
Run pip install -r requirements.txt to install necessary libraries, you may need to install more, just run and check for errors for missing libraries and install them<br/>
Run the CNNtrial.ipybn using jupyter notebook to train the CNN, the result is TrashCNN.h5 <br/>
Run waterroomba.py using python3 waterroomba.py to execute the main program, click on the displayframe and press q to quit <br/>
Check out demo: https://drive.google.com/file/d/1y0JDZ3-s8sQVQJbeDB_hYdMjbYDWbzVt/view?usp=sharing <br/><br/><br/>

![Alt text](./samplerunning.png?raw=true "Title")
