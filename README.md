This is a python implementation of the **User Equilibrium** and **Social Optimum** Traffic Assignment using the **Frank-Wolfe Algorithm**.
It uses the python-igraph library to calculate the all_or_nothing assignments in each iterations.
It includes implementation for an LA network and Chicago Regional Network. 
The implementation also allows to increase/decrease the demand by specifying a ratio.

For example to run the program on the LA network:<br />
**python Demand_Study.py LA 0.5 UE**<br />
The first argument "LA" specifies the name of the network, 0.5 means that the original demand will be halved, UE indicates that this is User Equilibrium case <br />

As another example:<br />
**python Demand_Study.py ChicagoRegional 1 SO**<br />
This command runs the Frank-Wolfe algorithm on the Chicago Regional Network, with the original demand (demand X 1), and looking for the Social Optimum Traffic Assignment.
