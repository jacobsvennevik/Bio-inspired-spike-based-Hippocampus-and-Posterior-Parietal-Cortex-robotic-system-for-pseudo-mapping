"""
PyNN+NEST implementation of a simple hippocampal associative memory.

This does NOT attempt to fully replicate the original sPyMem implementation,
but it exposes a compatible-enough surface for this project:

- A CA3-like \"content\" population that other modules (e.g. PPC) can read.
- An STDP-based projection from the input population into the CA3 content
  population so that patterns can be learned through repeated co-activation.

The driver script (`real_time_map_and_nav_pynn.py`) is responsible for
encoding grid-cell addresses and map-state codes into the input population.
"""

from dataclasses import dataclass
from typing import Any

import pyNN.nest as sim  # type: ignore


@dataclass
class HippocampusNetwork:
    """
    Lightweight container for the hippocampal network pieces we need.

    Attributes
    ----------
    input_pop:
        The input population that receives encoded (cue, state) activity.
    ca3_cont_pop:
        CA3-like content population representing learned map states.
    cue_to_cont_proj:
        STDP projection from input to CA3 content.
    """

    input_pop: Any
    ca3_cont_pop: Any
    cue_to_cont_proj: Any


def build_hippocampus(
    input_pop,
    cue_size: int,
    cont_size: int,
) -> HippocampusNetwork:
    """
    Build a very simple CA3-like associative memory using PyNN+NEST.

    Parameters
    ----------
    input_pop:
        PyNN Population providing spike input that encodes both \"cue\"
        (grid-cell address) and \"content\" (state code).
        The precise encoding is handled by the caller.
    cue_size:
        Number of distinct cue patterns (e.g. number of grid cells).
        Currently used only as a reference / sanity check.
    cont_size:
        Number of distinct content states (e.g. 8 map states).

    Returns
    -------
    HippocampusNetwork
        Container with the CA3 content population and the STDP projection.
    """

    # Basic LIF neuron parameters for CA3 content cells
    neuron_params = {
        "cm": 0.25,
        "tau_m": 20.0,
        "tau_refrac": 2.0,
        "tau_syn_E": 5.0,
        "tau_syn_I": 5.0,
        "v_rest": -65.0,
        "v_reset": -65.0,
        "v_thresh": -50.0,
        "i_offset": 0.0,
    }

    # CA3 content layer: one neuron per possible map state
    ca3_cont_pop = sim.Population(
        cont_size,
        sim.IF_curr_exp(**neuron_params),
        label="CA3contLayer_pynn",
    )

    # Simple all-to-all associative projection with STDP
    # The exact parameters are not critical for the current use case;
    # we use a standard pair-based STDP rule.
    timing_rule = sim.SpikePairRule(
        tau_plus=20.0,
        tau_minus=20.0,
        A_plus=0.01,
        A_minus=0.012,
    )
    weight_rule = sim.AdditiveWeightDependence(
        w_min=0.0,
        w_max=1.0,
    )
    stdp_mech = sim.STDPMechanism(
        timing_dependence=timing_rule,
        weight_dependence=weight_rule,
        weight=0.1,
        delay=1.0,
    )

    cue_to_cont_proj = sim.Projection(
        input_pop,
        ca3_cont_pop,
        sim.AllToAllConnector(),
        synapse_type=stdp_mech,
        receptor_type="excitatory",
    )

    # Record content spikes for later analysis / plotting
    ca3_cont_pop.record(["spikes"])

    return HippocampusNetwork(
        input_pop=input_pop,
        ca3_cont_pop=ca3_cont_pop,
        cue_to_cont_proj=cue_to_cont_proj,
    )



