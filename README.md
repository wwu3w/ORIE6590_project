# ORIE6590_project
state space:
car state: 2D array with dimention Rx(tau_d + L). Each entry represents the number of cars that need time tau to reach the destination r.
passenger/order state: 2D array with dimention RxR. Each entry represents that number of orders that start from o to d.
city time: a integer that represents the cumulative running time of the system to up now

Action space: 2D array with dimension RxR. Each action specifiying trip from o to d. Each trip can be a car-passenger matching, car empty routing or do-nothing (o=d).

Rewards: If a car-passenger matching is successful, then a postive reward is marked as +1. Otherwise, there will be zero reward.

Transitions:
car state transition:
1). At each time step, each car is one time step closer to its destination.
2). If an atomic action is applied on a car, then its destination will be changed correspondingly.

passenger/order state transition:
1). New orders will appear at each grid according to the independent Poission distributions with pre-specified arrival rates
2). If a car-passenger matching (o, d) is successful, then the number of the orders at (o, d) is deducted by 1.


