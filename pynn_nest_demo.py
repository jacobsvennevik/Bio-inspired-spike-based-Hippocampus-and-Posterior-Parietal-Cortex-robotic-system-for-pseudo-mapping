import pyNN.nest as sim
import matplotlib.pyplot as plt


def main():
    # 1. Setup simulator
    sim.setup(timestep=0.1)  # ms

    # 2. Populations: Poisson input -> LIF neurons
    input_pop = sim.Population(
        10,
        sim.SpikeSourcePoisson(rate=20.0),
        label="poisson_input",
    )

    cell_params = {
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
    exc_pop = sim.Population(
        5,
        sim.IF_curr_exp(**cell_params),
        label="exc_neurons",
    )

    # 3. Projections: connect input to excitatory neurons
    connector = sim.FixedProbabilityConnector(p_connect=0.5)
    synapse = sim.StaticSynapse(weight=0.01, delay=1.0)  # nA, ms
    sim.Projection(input_pop, exc_pop, connector, synapse_type=synapse)

    # 4. Recording
    exc_pop.record(["spikes", "v"])

    # 5. Run simulation
    sim_time = 200.0  # ms
    sim.run(sim_time)

    # 6. Retrieve data
    data = exc_pop.get_data()
    segment = data.segments[0]
    spiketrains = segment.spiketrains
    v_traces = segment.filter(name="v")[0]

    # 7. Plot spikes and one voltage trace
    fig, (ax_raster, ax_v) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i, train in enumerate(spiketrains):
        ax_raster.vlines(train, i + 0.5, i + 1.5)
    ax_raster.set_ylabel("Neuron index")
    ax_raster.set_title("Spike raster (excitatory population)")

    ax_v.plot(v_traces.times, v_traces.magnitude[:, 0])
    ax_v.set_ylabel("V (mV)")
    ax_v.set_xlabel("Time (ms)")
    ax_v.set_title("Membrane potential of neuron 0")

    plt.tight_layout()
    plt.show()

    # 8. Clean up
    sim.end()


if __name__ == "__main__":
    main()


