from sPyBlocks.connection_functions import create_connections, inverse_rcp_type, multiple_connect, flatten


class NeuralSyncOscillator:
    """
    This class defines the Synchronous Oscillator block.
    """
    def __init__(self, n_period, sim, global_params, neuron_params, std_conn):
        """
        Constructor of the class.

        :param int n_period: The temporal length (duration) of the spike train.
        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the block. Commonly, its weight is 1.0 and its delay is equal to the timestep.
        """
        # Storing parameters
        self.n_period = n_period
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
        self.set_source = sim.Population(1, sim.SpikeSourceArray(spike_times=range(1, 1 + n_period)))
        self.input_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                           initial_values={'v': neuron_params["v_rest"]})
        self.output_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                            initial_values={'v': neuron_params["v_rest"]})

        self.total_neurons += self.set_source.size + self.input_neuron.size + self.output_neuron.size

        # Create the connections
        created_connections = 0

        created_connections += create_connections(self.set_source, self.input_neuron, sim, std_conn)

        delayed_conn = sim.StaticSynapse(weight=1.0, delay=n_period)
        created_connections += create_connections(self.input_neuron, self.output_neuron, self.sim, delayed_conn)
        created_connections += create_connections(self.output_neuron, self.input_neuron, self.sim, delayed_conn)

        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = self.std_conn.delay + n_period

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None):
        """
        Connects the output neurons of the block to an output population.

        :param sim.Population, sim.PopulationView, sim.Assembly, list output_population: A PyNN object or a list of PyNN objects containing the population to connect the output neurons to.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: Unused.
        :param end_pop_indexes: A list of indices used to select objects from the output population.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(self.output_neuron, output_population, self.sim, conn,
                                                 rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_output_neuron(self, flat=False):
        """
        Gets a list containing all the output neurons of the block.

        :param bool flat: Unused.
        :return: The list containing all the output neurons of the block
        :rtype: list
        """
        return [self.output_neuron]


class NeuralAsyncOscillator:
    """
    This class defines the Asynchronous Oscillator (Switch) block.
    """
    def __init__(self, sim, global_params, neuron_params, std_conn):
        """
        Constructor of the class.

        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the block. Commonly, its weight is 1.0 and its delay is equal to the timestep.
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
        self.input_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                           initial_values={'v': neuron_params["v_rest"]})
        self.cycle_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                           initial_values={'v': neuron_params["v_rest"]})

        self.total_neurons += self.input_neuron.size + self.cycle_neuron.size

        # Create the connections
        created_connections = 0

        created_connections += create_connections(self.input_neuron, self.cycle_neuron, sim, std_conn)
        created_connections += create_connections(self.cycle_neuron, self.input_neuron, sim, std_conn,
                                                  rcp_type="inhibitory")

        created_connections += create_connections(self.input_neuron, self.input_neuron, sim, std_conn,
                                                  rcp_type="inhibitory")
        created_connections += create_connections(self.cycle_neuron, self.cycle_neuron, sim, std_conn)

        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = 0

    def connect_signal(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        """
        Connects an input population, whose spikes indicate when to switch the internal value of the block, to the input neurons of the block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the input neurons.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: Unused.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        inv_rcp_type = inverse_rcp_type(rcp_type)

        created_connections = 0
        created_connections += create_connections(input_population, self.input_neuron, self.sim, conn,
                                                  rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)
        created_connections += create_connections(input_population, self.cycle_neuron, self.sim, conn,
                                                  rcp_type=inv_rcp_type, ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None):
        """
        Connects the output neurons of the block to an output population.

        :param sim.Population, sim.PopulationView, sim.Assembly, list output_population: A PyNN object or a list of PyNN objects containing the population to connect the output neurons to.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: Unused.
        :param end_pop_indexes: A list of indices used to select objects from the output population.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        created_connections = 0
        created_connections += create_connections(self.input_neuron, output_population, self.sim, conn,
                                                  rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)
        created_connections += create_connections(self.cycle_neuron, output_population, self.sim, conn,
                                                  rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_signal_neurons(self, flat=False):
        """
        Gets a list containing all the input (signal) neurons of the block.

        :param bool flat: Unused.
        :return: The flattened or unflattened list containing all the input (signal) neurons of the block
        :rtype: list
        """
        return [self.input_neuron, self.cycle_neuron]

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the block.

        :param bool flat: Unused.
        :return: The flattened or unflattened list containing all the output neurons of the block
        :rtype: list
        """
        return [self.input_neuron, self.cycle_neuron]


class MultipleNeuralAsyncOscillator:
    """
    This class allows to create multiple Asynchronous Oscillator (Switch) blocks of the same type.
    """
    def __init__(self, n_components, sim, global_params, neuron_params, std_conn):
        """
        Constructor of the class.

        :param int n_components: The number of blocks to create.
        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the blocks. Commonly, its weight is 1.0 and its delay is equal to the timestep.
        """
        # Storing parameters
        self.n_components = n_components
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the array of multiple src
        self.oscillator_array = []
        for i in range(n_components):
            switch = NeuralAsyncOscillator(sim, global_params, neuron_params, std_conn)
            self.oscillator_array.append(switch)

            self.total_neurons += switch.total_neurons
            self.total_internal_connections += switch.total_internal_connections

        # Total internal delay
        self.delay = self.oscillator_array[0].delay

    def connect_signal(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, component_indexes=None):
        """
        Connects an input population, whose spikes indicate when to switch the internal value of the blocks, to the input (signal) neurons of the block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the input (signal) neurons.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: Unused.
        :param list component_indexes: A list of indices used to select components from the list of components.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_signal", input_population, self.oscillator_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, component_indexes=None):
        """
        Connects the output neurons of the blocks to an output population.

        :param sim.Population, sim.PopulationView, sim.Assembly, list output_population: A PyNN object or a list of PyNN objects containing the population to connect the output neurons to.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: Unused.
        :param end_pop_indexes: A list of indices used to select objects from the output population.
        :param list component_indexes: A list of indices used to select components from the list of components.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_outputs", output_population, self.oscillator_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_signal_neurons(self, flat=False):
        """
        Gets a list containing all the input (signal) neurons of the blocks.

        :param bool flat: Unused.
        :return: The flattened or unflattened list containing all the input (signal) neurons of the blocks
        :rtype: list
        """
        input_neurons = []

        for i in range(self.n_components):
            input_neurons.append(self.oscillator_array[i].get_signal_neurons())

        if flat:
            return flatten(input_neurons)
        else:
            return input_neurons

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the blocks.

        :param bool flat: Unused.
        :return: The flattened or unflattened list containing all the output neurons of the blocks
        :rtype: list
        """
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.oscillator_array[i].get_output_neurons())

        if flat:
            return flatten(output_neurons)
        else:
            return output_neurons
