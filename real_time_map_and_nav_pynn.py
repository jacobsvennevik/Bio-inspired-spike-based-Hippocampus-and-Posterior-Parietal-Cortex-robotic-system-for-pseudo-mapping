"""
Offline PyNN+NEST version of the real-time map and navigation app.

Key differences from the original SpiNNaker implementation:
    - Uses `pyNN.nest` as the backend (no spynnaker8, no live I/O).
    - Navigation and map updates are performed in Python in a synchronous loop.
    - A simple hippocampal memory and PPC timing network are built for
      spike-based analysis, but the high-level path planning logic mirrors
      the original Python code.

The script writes its results under `results/<experimentName>_pynn/`:
    - initial_map_formatted.txt, initial_map.txt
    - final_map_formatted.txt, final_map.txt
    - out_mem_spikes.txt (CA3 content population)
    - out_ppc_spikes.txt (PPC output layer)
"""

import collections
import collections.abc
import math
import os
from typing import List, Tuple

import numpy as np
import pyNN.nest as sim  # type: ignore

from hippocampus_pynn import build_hippocampus
from ppc_pynn import build_ppc


# Compatibility shims for older dependencies (neo, PyNN) on modern Python.
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence  # type: ignore[attr-defined]
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]
if not hasattr(collections, "MutableSet"):
    collections.MutableSet = collections.abc.MutableSet  # type: ignore[attr-defined]
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]


#############################################
# Experiment configuration
#############################################


# Choose the desired experiment (match original IDs):
#   + 0  -> Robot demo: test 1 with a real robot (not used here)
#   + 1  -> Test 1: 4x4 map with 1 obstacle in the path
#   + 2  -> Test 2: 6x6 map with various obstacle in the map
#   + 3  -> Test 3: 6x6 map with various obstacle in the map blocking
#                   the manhattan possible paths
experiment = 2


if experiment == 0:
    # Robot demo parameters (kept for completeness, not used here)
    xlength = 4
    ylength = 4
    xinit = 2
    yinit = 0
    xend = 0
    yend = 3
    experimentName = "robotDemo"
    obstacles: List[int] = []

elif experiment == 1:
    # Test 1: 4x4 map with 1 obstacle in the path
    xlength = 4
    ylength = 4
    xinit = 2
    yinit = 0
    xend = 0
    yend = 3
    experimentName = "test4x4simple_pynn"
    obstacles = [11]

elif experiment == 2:
    # Test 2: 6x6 map with obstacles
    xlength = 6
    ylength = 6
    xinit = 2
    yinit = 0
    xend = 5
    yend = 5
    experimentName = "test6x6simple_pynn"
    obstacles = [5, 12, 15, 23, 30, 34]

else:
    # Test 3: 6x6 map with more complex obstacle configuration
    xlength = 6
    ylength = 6
    xinit = 2
    yinit = 0
    xend = 5
    yend = 5
    experimentName = "test6x6complex_pynn"
    obstacles = [5, 9, 12, 15, 21, 23, 28, 30]


#############################################
# Global-like parameters (mirroring original)
#############################################


numStates = 8  # number of states of the map state
debugLevel = 1
write_results = True

# How many repeated iterations of searching cells is considered a dead end
maxRepeatedIteration = 3

# Time parameters for PPC-like timing in ms
timeStep = 1.0
operationTime = 0.03  # s, approximated from original


def manhattan_nearest_cell_to_target(target: Tuple[int, int], cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Get list of cells with the nearest distance to the target with manhattan distance."""
    nearest_cells: List[Tuple[int, int]] = []
    nearest_distance = -1
    for cell in cells:
        distance = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
        if nearest_distance == -1 or distance <= nearest_distance:
            if distance == nearest_distance:
                nearest_cells.append(cell)
            else:
                nearest_cells = [cell]
            nearest_distance = distance
    return nearest_cells


def check_folder_and_create_file(data, path: str, fileName: str) -> None:
    """Write data in file and create path if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, fileName), "w") as f:
        f.write(str(data))


def direction_from_cells(
    last_cell: Tuple[int, int],
    next_cell: Tuple[int, int],
) -> int:
    """
    Compute movement command from last->next cell.

    Commands follow the original convention:
        0 = top, 1 = left, 2 = right, 3 = bottom
    """
    (cx, cy) = last_cell
    (nx, ny) = next_cell
    if ny == cy - 1 and nx == cx:
        return 0  # top
    if nx == cx - 1 and ny == cy:
        return 1  # left
    if nx == cx + 1 and ny == cy:
        return 2  # right
    if ny == cy + 1 and nx == cx:
        return 3  # bottom
    raise ValueError(f"Non-adjacent move from {last_cell} to {next_cell}")


def build_network(numInputLayerNeurons: int):
    """
    Build the PyNN hippocampus + PPC network.

    Returns
    -------
    (hippo, ppc_out_pop, search_pop, match_pop)
    """

    # Input population for hippocampus: driver can encode arbitrary spike trains
    input_pop = sim.Population(
        numInputLayerNeurons,
        sim.SpikeSourceArray(spike_times=[[] for _ in range(numInputLayerNeurons)]),
        label="InputLayer",
    )

    # Simple hippocampus network
    hippo = build_hippocampus(
        input_pop=input_pop,
        cue_size=xlength * ylength,
        cont_size=numStates,
    )

    # PPC network: 1-neuron search trigger + 4-neuron match (one per command)
    search_pop = sim.Population(
        1,
        sim.SpikeSourceArray(spike_times=[[]]),
        label="SearchingINLayer",
    )
    match_pop = sim.Population(
        4,
        sim.SpikeSourceArray(spike_times=[[] for _ in range(4)]),
        label="MatchINLayer",
    )

    from ppc_pynn import build_ppc as _build_ppc

    ppc, ppc_out_pop = _build_ppc(
        sim=sim,
        search_pop=search_pop,
        match_pop=match_pop,
        num_commands=4,
        operation_delay_ms=operationTime * 1000.0,
        initial_delay_ms=9.0,
    )

    # Record CA3 content (for out_mem_spikes) and PPC outputs
    hippo.ca3_cont_pop.record(["spikes"])
    ppc_out_pop.record(["spikes"])

    return hippo, ppc_out_pop, search_pop, match_pop


def run_navigation_loop() -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[List[float]]]:
    """
    Run the synchronous navigation loop using a Python implementation of the
    original neighbour / backtracking logic. The hippocampus and PPC networks
    are used to generate spike data in parallel with the navigation steps.
    """

    # Network-encoding parameters (match original as far as needed)
    cueSize = xlength * ylength
    contSize = numStates
    cueSizeInBin = math.ceil(math.log2(cueSize + 1))
    numInputLayerNeurons = cueSizeInBin + contSize

    # Build PyNN network
    sim.setup(timeStep)
    hippo, ppc_out_pop, search_pop, match_pop = build_network(numInputLayerNeurons)

    # Initial map
    initial_map_state = np.zeros((ylength, xlength), dtype=int)
    initial_map_state[yinit][xinit] = 1
    initial_map_state[yend][xend] = 2

    # Final map (we update it directly here instead of via memory sweep)
    final_map_state = initial_map_state.copy()

    # Navigation state
    cellX = xinit
    cellY = yinit
    lastCellX = xinit
    lastCellY = yinit
    robotPath: List[int] = [yinit * xlength + xinit + 1]

    backtracking = False
    crossroadCell = False
    nearestCell = (xinit, yinit)
    iterationsRepeated = 0
    finish = False
    unachievable = False

    # Spike scheduling helpers
    # Start strictly after t=0.0 to avoid NEST complaining about spike time 0
    current_time = 1.0
    chunk_ms = 50.0
    search_spike_times: List[float] = []
    match_spike_times = [[] for _ in range(4)]  # one list per command neuron

    target = (xend, yend)

    while not finish and not unachievable:
        # 1) Determine free neighbours and obstacles (Python-only, similar to check_real_neighbours)
        freeCells: List[Tuple[int, int]] = []
        obstacleCells: List[Tuple[int, int]] = []

        for direction in range(4):
            nx, ny = cellX, cellY
            if direction == 0 and cellY - 1 >= 0:
                ny = cellY - 1
            elif direction == 1 and cellX - 1 >= 0:
                nx = cellX - 1
            elif direction == 2 and cellX + 1 < xlength:
                nx = cellX + 1
            elif direction == 3 and cellY + 1 < ylength:
                ny = cellY + 1
            else:
                continue

            position = ny * xlength + nx + 1
            if position in obstacles:
                obstacleCells.append((nx, ny))
                final_map_state[ny, nx] = 6  # obstacle
            elif position not in robotPath and not backtracking:
                freeCells.append((nx, ny))
                # mark as free if not already a special state
                if final_map_state[ny, nx] == 0:
                    final_map_state[ny, nx] = 5  # free

        # 2) If there is a known next step in the path, follow it; otherwise pick a new one
        if freeCells:
            nearestCells = manhattan_nearest_cell_to_target(target, freeCells)
            nearestCell = nearestCells[-1]
            nextCellX, nextCellY = nearestCell
            # Mark current cell as part of path
            if final_map_state[cellY, cellX] not in (1, 2):
                final_map_state[cellY, cellX] = 3  # step in path
        else:
            # Dead end: backtracking would normally occur; here we simply stop if repeated
            iterationsRepeated += 1
            if iterationsRepeated >= maxRepeatedIteration:
                unachievable = True
                break
            backtracking = True
            # simple backtracking: move to previous cell in path
            if len(robotPath) > 1:
                robotPath.pop()
                last_id = robotPath[-1]
                nextCellX = (last_id - 1) % xlength
                nextCellY = (last_id - 1) // xlength
            else:
                unachievable = True
                break

        # 3) Compute command and update path state
        command = direction_from_cells((cellX, cellY), (nextCellX, nextCellY))
        lastCellX, lastCellY = cellX, cellY
        cellX, cellY = nextCellX, nextCellY
        robotPath.append(cellY * xlength + cellX + 1)

        # Mark crossroads and dead ends roughly as in the original
        if len(freeCells) > 1:
            final_map_state[lastCellY, lastCellX] = 4  # crossroad

        # Check if we reached target
        if cellX == xend and cellY == yend:
            final_map_state[cellY, cellX] = 2  # end
            finish = True
        else:
            if final_map_state[cellY, cellX] not in (1, 2):
                final_map_state[cellY, cellX] = 3

        # 4) Encode this command into the PPC network as spikes
        # Ensure all spike times are strictly > 0.0 for NEST
        eps = 1e-3
        search_spike_times.append(max(current_time + 1.0, eps))
        match_spike_times[command].append(max(current_time + 10.0, eps))

        safe_search_times = [t if t > 0.0 else eps for t in search_spike_times]
        safe_match_times = [
            [t if t > 0.0 else eps for t in neuron_times]
            for neuron_times in match_spike_times
        ]

        search_pop.set(spike_times=[safe_search_times])
        match_pop.set(spike_times=safe_match_times)

        # Run a short simulation chunk to generate spikes
        sim.run(chunk_ms)
        current_time += chunk_ms

        # Loop termination guards
        iterationsRepeated += 1
        if iterationsRepeated >= 200:  # safety cap
            unachievable = True

    # End simulation and collect spikes
    mem_data = hippo.ca3_cont_pop.get_data(variables=["spikes"]).segments[0].spiketrains
    out_mem_spikes: List[List[float]] = [st.as_array().tolist() for st in mem_data]

    ppc_data = ppc_out_pop.get_data(variables=["spikes"]).segments[0].spiketrains
    out_ppc_spikes: List[List[float]] = [st.as_array().tolist() for st in ppc_data]

    sim.end()

    return initial_map_state, final_map_state, out_mem_spikes, out_ppc_spikes


def main() -> None:
    # Run navigation and collect results
    initial_map_state, final_map_state, out_mem_spikes, out_ppc_spikes = run_navigation_loop()

    # Prepare output folder
    filePath = os.path.join("results", experimentName)

    if write_results:
        check_folder_and_create_file(initial_map_state, filePath, "initial_map_formatted.txt")
        check_folder_and_create_file(initial_map_state.tolist(), filePath, "initial_map.txt")
        check_folder_and_create_file(final_map_state, filePath, "final_map_formatted.txt")
        check_folder_and_create_file(final_map_state.tolist(), filePath, "final_map.txt")
        check_folder_and_create_file(out_mem_spikes, filePath, "out_mem_spikes.txt")
        check_folder_and_create_file(out_ppc_spikes, filePath, "out_ppc_spikes.txt")


if __name__ == "__main__":
    main()



