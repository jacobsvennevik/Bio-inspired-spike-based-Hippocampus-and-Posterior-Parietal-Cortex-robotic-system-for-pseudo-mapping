from sPyBlocks.connection_functions import create_connections, flatten
from sPyBlocks.neural_latch_sr import NeuralLatchSR


class ConstantSpikeSource:
    """
    This class defines the Constant Spike Source (CSS) block, a block which constantly fires spikes. Its use allows the
    correct operation of the NOT and fast AND gates.
    """
    def __init__(self, sim, global_params, neuron_params, std_conn):
        """
        Constructor of the class.

        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the block. Commonly, its weight is 1.0 and its delay is equal to the timestep. Using other values could change the behavior of the block.
        """
        # Storing parameters
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        self.set_source = sim.Population(1, sim.SpikeSourceArray(spike_times=[1.0]))
        self.latch = NeuralLatchSR(sim, global_params, neuron_params, std_conn)

        self.total_neurons += self.set_source.size + self.latch.total_neurons
        self.total_internal_connections += self.latch.total_internal_connections

        # Create the connections
        created_connections = self.latch.connect_set(self.set_source)
        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = 0

    def connect_outputs(self, output_population, conn=None, rcp_type="excitatory"):
        """
        Connects the output neurons of the block to an output population.

        :param sim.Population, sim.PopulationView, sim.Assembly, list output_population: A PyNN object or a list of PyNN objects containing the population to connect the output neurons to.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        created_connections = 0

        created_connections += create_connections(self.set_source, output_population, self.sim, conn, rcp_type=rcp_type)
        created_connections += self.latch.connect_output(output_population)

        self.total_output_connections += created_connections
        return created_connections

    def get_output_neuron(self, flat=False):
        """
        Gets a list containing all the output neurons of the block

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The flattened or unflattened list containing all the output neurons of the block
        :rtype: list
        """
        if flat:
            return flatten([self.set_source, self.latch.get_output_neurons()])
        else:
            return [self.set_source, self.latch.get_output_neurons()]