"""
PyNN+NEST wrapper for the PPC (Posterior Parietal Cortex) timing network.

This module refactors the original `posterior_parietal_cortex.PPC` so that it
can be used with any PyNN-compatible backend, in particular `pyNN.nest`.

The network:
- Receives a 1-neuron \"search\" trigger population.
- Receives a small \"match\" population (typically driven by hippocampus).
- Produces a 4-neuron command output population over time.

The exact dynamics follow the original PPC implementation.
"""

from typing import Any, Dict, Optional, Tuple


class PPC:
    def __init__(
        self,
        SearchingINLayer,
        MatchINLayer,
        OLayer,
        numCommands: int,
        operationDelay: float,
        initialDelay: float,
        sim,
    ):
        """Constructor method.

        Parameters
        ----------
        SearchingINLayer:
            1-neuron population that provides the \"search\" trigger.
        MatchINLayer:
            Small population providing match signals (typically from hippocampus).
        OLayer:
            Output command population (numCommands neurons).
        numCommands:
            Number of distinct commands / time slots.
        operationDelay:
            Delay (ms) between successive commands in the delay chain.
        initialDelay:
            Initial delay (ms) before the first command becomes active.
        sim:
            PyNN simulator module (e.g. `pyNN.nest`).
        """

        # Storing parameters
        self.SearchingINLayer = SearchingINLayer
        self.MatchINLayer = MatchINLayer
        self.OLayer = OLayer
        self.sim = sim
        self.numCommands = numCommands
        self.operationDelay = operationDelay
        self.initialDelay = initialDelay

        # Create the network
        self.create_population()
        self.create_synapses()

    def create_population(self) -> None:
        """Create all populations of the PPC model."""

        # Define neurons parameters
        self.neuronParameters: Dict[str, float] = {
            "cm": 0.27,
            "i_offset": 0.0,
            "tau_m": 3.0,
            "tau_refrac": 1.0,
            "tau_syn_E": 0.3,
            "tau_syn_I": 0.3,
            "v_reset": -60.0,
            "v_rest": -60.0,
            "v_thresh": -57.0,
        }
        # ReadLayer
        self.InitDelayLayer = self.sim.Population(
            1,
            self.sim.IF_curr_exp(**self.neuronParameters),
            label="SearchingLayer",
        )

        # DelayLayer
        self.DelayLayer = self.sim.Population(
            self.numCommands,
            self.sim.IF_curr_exp(**self.neuronParameters),
            label="DelayLayer",
        )

        # MatchLayer
        self.MatchLayer = self.sim.Population(
            self.numCommands,
            self.sim.IF_curr_exp(**self.neuronParameters),
            label="MatchLayer",
        )

        # InhLayer
        self.InhLayer = self.sim.Population(
            1,
            self.sim.IF_curr_exp(**self.neuronParameters),
            label="InhLayer",
        )

    def create_synapses(self) -> None:
        """Create all synapses of the PPC model."""

        # SearchingINLayer-InitDelayLayer -> 1 to 1, excitatory and static
        self.SearchingINL_InitDelayL = self.sim.Projection(
            self.SearchingINLayer,
            self.InitDelayLayer,
            self.sim.OneToOneConnector(),
            synapse_type=self.sim.StaticSynapse(weight=6.0, delay=1.0),
            receptor_type="excitatory",
        )

        # InitDelayLayer-DelayLayer -> 1 to 1 (first neuron), excitatory and static
        self.InitDelayL_DelayL = self.sim.Projection(
            self.InitDelayLayer,
            self.sim.PopulationView(self.DelayLayer, [0]),
            self.sim.OneToOneConnector(),
            synapse_type=self.sim.StaticSynapse(
                weight=6.0,
                delay=self.initialDelay,
            ),
            receptor_type="excitatory",
        )

        # DelayLayer-DelayLayer -> 1 to 1 (i -> i+1), excitatory and static
        for neuronId in range(self.numCommands - 1):
            self.sim.Projection(
                self.sim.PopulationView(self.DelayLayer, [neuronId]),
                self.sim.PopulationView(self.DelayLayer, [neuronId + 1]),
                self.sim.OneToOneConnector(),
                synapse_type=self.sim.StaticSynapse(
                    weight=6.0,
                    delay=self.operationDelay,
                ),
                receptor_type="excitatory",
            )

        # DelayLayer-MatchLayer -> 1 to 1, excitatory and static
        self.DelayL_MatchL = self.sim.Projection(
            self.DelayLayer,
            self.MatchLayer,
            self.sim.OneToOneConnector(),
            synapse_type=self.sim.StaticSynapse(weight=2.5, delay=1.0),
            receptor_type="excitatory",
        )

        # MatchINLayer-MatchLayer -> all to 1 (for each), excitatory and static
        for neuronId in range(self.numCommands):
            self.sim.Projection(
                self.MatchINLayer,
                self.sim.PopulationView(self.MatchLayer, [neuronId]),
                self.sim.AllToAllConnector(allow_self_connections=True),
                synapse_type=self.sim.StaticSynapse(weight=2.5, delay=1.0),
                receptor_type="excitatory",
            )

        # MatchLayer-OLayer -> 1 to 1, excitatory and static
        self.MatchL_OL = self.sim.Projection(
            self.MatchLayer,
            self.OLayer,
            self.sim.OneToOneConnector(),
            synapse_type=self.sim.StaticSynapse(weight=6.0, delay=1.0),
            receptor_type="excitatory",
        )

        # DelayLayer-InhLayer
        self.DelayL_InhL = self.sim.Projection(
            self.sim.PopulationView(self.DelayLayer, [self.numCommands - 1]),
            self.InhLayer,
            self.sim.OneToOneConnector(),
            synapse_type=self.sim.StaticSynapse(weight=6.0, delay=1.0),
            receptor_type="excitatory",
        )

        # InhLayer-MatchLayer
        self.InhL_MatchL = self.sim.Projection(
            self.InhLayer,
            self.MatchLayer,
            self.sim.AllToAllConnector(),
            # Inhibitory synapses must have NEGATIVE weights in PyNN+NEST.
            synapse_type=self.sim.StaticSynapse(weight=-6.0, delay=1.0),
            receptor_type="inhibitory",
        )


def build_ppc(
    sim,
    search_pop,
    match_pop,
    num_commands: int = 4,
    operation_delay_ms: float = 7.0,
    initial_delay_ms: float = 9.0,
    neuron_params: Optional[Dict[str, float]] = None,
) -> Tuple[PPC, Any]:
    """
    Convenience builder for the PPC network using PyNN+NEST.

    Parameters
    ----------
    sim:
        PyNN simulator module (e.g. `pyNN.nest`).
    search_pop:
        1-neuron population used as searching trigger.
    match_pop:
        Small population of SpikeSource / neurons driven by hippocampus.
    num_commands:
        Number of commands and output neurons.
    operation_delay_ms:
        Delay between successive commands in the delay chain.
    initial_delay_ms:
        Initial delay before the first command.
    neuron_params:
        Optional neuron parameter override for the output layer.

    Returns
    -------
    (ppc, output_pop)
        The PPC instance and its output command population.
    """

    if neuron_params is None:
        neuron_params = {
            "cm": 0.27,
            "i_offset": 0.0,
            "tau_m": 3.0,
            "tau_refrac": 1.0,
            "tau_syn_E": 0.3,
            "tau_syn_I": 0.3,
            "v_reset": -60.0,
            "v_rest": -60.0,
            "v_thresh": -57.5,
        }

    OLayer = sim.Population(
        num_commands,
        sim.IF_curr_exp(**neuron_params),
        label="OPPCLayer",
    )

    ppc = PPC(
        SearchingINLayer=search_pop,
        MatchINLayer=match_pop,
        OLayer=OLayer,
        numCommands=num_commands,
        operationDelay=operation_delay_ms,
        initialDelay=initial_delay_ms,
        sim=sim,
    )

    # Let the caller decide what to record; commonly we record OLayer spikes.
    return ppc, OLayer



