4x4 SpiNNaker:

[[0 5 4 5]
 [5 4 4 5]
 [5 4 6 0]
 [2 3 5 0]]

PyNN+NEST final map:
 [[0 5 4 5]
 [5 4 4 5]
 [5 4 6 0]
 [2 4 5 0]]

Very very similar only difference is bottom row second column is 4 instead of 3. Has to with crossroads. Other than that identical!

6x6 SpiNNaker:
[[0 5 4 3 6 0]
 [0 0 5 4 3 6]
 [0 0 6 3 3 7]
 [0 0 5 4 6 7]
 [0 0 5 4 3 6]
 [0 0 0 6 3 2]]

 PyNN+NEST final map:
 [[0 5 4 5 0 0]
 [0 5 4 4 5 0]
 [0 0 6 4 5 0]
 [0 0 5 4 6 0]
 [0 0 5 4 3 6]
 [0 0 0 6 3 2]]

 Seems like same target is reached, but slightly different in marking intermediate cells. PyNN+NEST is more conservative.