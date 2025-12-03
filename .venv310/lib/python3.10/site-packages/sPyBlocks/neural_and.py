from .connection_functions import create_connections, multiple_connect, flatten
from .neural_or import NeuralOr


class NeuralAnd:
    """
    This class defines the AND block. This block has two variants: classic and fast.
    """
    def __init__(self, n_inputs, sim, global_params, neuron_params, std_conn, build_type="classic"):
        """
        Constructor of the class.

        :param int n_inputs: The number of expected inputs of the block.
        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the block. Commonly, its weight is 1.0 and its delay is equal to the timestep. Using other values could change the behavior of the block.
        :param str build_type: A string indicating the AND variant ("classic" or "fast"). "classic" by default.
        :raise ValueError: If the build_type string is not "classic" or "fast".
        """
        # Storing parameters
        self.n_inputs = n_inputs
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        if build_type == "classic" or build_type == "fast":
            self.build_type = build_type
        else:
            raise ValueError("This build type is not implemented.")

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        if build_type == "classic":
            self.or_gate = NeuralOr(sim, global_params, neuron_params, std_conn)

            self.total_neurons += self.or_gate.total_neurons
            self.total_internal_connections += self.or_gate.total_internal_connections

        self.output_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                            initial_values={'v': neuron_params["v_rest"]})
        self.total_neurons += self.output_neuron.size

        # Custom synapses
        self.inh_synapse = sim.StaticSynapse(weight=n_inputs - 1, delay=global_params["min_delay"])

        # Create the connections
        if build_type == "classic":
            created_connections = create_connections(self.or_gate.output_neuron, self.output_neuron, sim,
                                                     self.inh_synapse, rcp_type="inhibitory")

            self.total_internal_connections += created_connections

        # Total internal delay
        if build_type == "classic":
            self.delay = self.or_gate.delay + self.inh_synapse.delay
        else:
            self.delay = 0

    def connect_inhibition(self, input_population, conn=None, conn_all=True, rcp_type="inhibitory",
                           ini_pop_indexes=None, end_pop_indexes=None):
        """
        Connects an input population to the neurons to be inhibited by the Constant Spike Source block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the inhibited neurons.
        :param sim.StaticSynapse conn: The connection to use. Internal inhibitory synapse by default, it is recommended NOT to use this parameter.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Inhibitory" by default, it is recommended NOT to use this parameter.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: Unused.
        :return: The number of connections that have been created.
        :rtype: int
        :raise ValueError: If this functions is used with the classic AND variant.
        """
        if self.build_type == "fast":
            if conn is None:
                conn = self.inh_synapse

            created_connections = create_connections(input_population, self.output_neuron, self.sim, conn,
                                                     rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

            self.total_input_connections += created_connections
            return created_connections
        else:
            raise TypeError("connect_inhibition function is not allowed for Classic AND gates")

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        """
        Connects an input population to the input neurons of the block.

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

        created_connections = 0
        if self.build_type == "classic":
            created_connections += self.or_gate.connect_inputs(input_population, conn, rcp_type=rcp_type,
                                                               ini_pop_indexes=ini_pop_indexes)

        delayed_conn = self.sim.StaticSynapse(weight=conn.weight, delay=conn.delay + self.delay)
        created_connections += create_connections(input_population, self.output_neuron, self.sim, delayed_conn,
                                                  rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

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

        created_connections = create_connections(self.output_neuron, output_population, self.sim, conn,
                                                 rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_inhibited_neuron(self, flat=False):
        """
        Gets a list containing all the neurons inhibited by the Constant Spike Source block.

        :param bool flat: Unused.
        :return: The list containing all the inhibited neurons of the block
        :rtype: list
        """
        if self.build_type == "classic":
            return []
        else:
            return [self.output_neuron]

    def get_input_neurons(self, flat=False):
        """
        Gets a list containing all the input neurons of the block.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The flattened or unflattened list containing all the input neurons of the block
        :rtype: list
        """
        if self.build_type == "classic":
            if flat:
                return flatten([self.or_gate.get_output_neuron(), self.output_neuron])
            else:
                return [self.or_gate.get_output_neuron(), self.output_neuron]
        else:
            return [self.output_neuron]

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the block.

        :param bool flat: Unused.
        :return: The list containing all the output neurons of the block
        :rtype: list
        """
        return [self.output_neuron]


class MultipleNeuralAnd:
    """
    This class allows to create multiple AND blocks of the same type.
    """
    def __init__(self, n_components, n_inputs, sim, global_params, neuron_params, std_conn, build_type="classic"):
        """
        Constructor of the class.

        :param int n_components: The number of blocks to create.
        :param int n_inputs: The number of expected inputs of the blocks.
        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the blocks. Commonly, its weight is 1.0 and its delay is equal to the timestep. Using other values could change the behavior of the block.
        :param str build_type: A string indicating the AND variant ("classic" or "fast"). "classic" by default.
        :raise ValueError: If the build_type string is not "classic" or "fast".
        """
        # Storing parameters
        self.n_components = n_components
        self.n_inputs = n_inputs
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        if build_type == "classic" or build_type == "fast":
            self.build_type = build_type
        else:
            raise ValueError("This type of AND gate is not implemented.")

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the array of multiple src
        self.and_array = []
        for i in range(n_components):
            and_gate = NeuralAnd(n_inputs, sim, global_params, neuron_params, std_conn, build_type=build_type)
            self.and_array.append(and_gate)

            self.total_neurons += and_gate.total_neurons
            self.total_internal_connections += and_gate.total_internal_connections

        # Custom synapses
        self.inh_synapse = self.and_array[0].inh_synapse

        # Total internal delay
        self.delay = self.and_array[0].delay

    def connect_inhibition(self, input_population, conn=None, conn_all=True, rcp_type="inhibitory",
                           ini_pop_indexes=None, end_pop_indexes=None, component_indexes=None):
        """
        Connects an input population to the neurons to be inhibited by the Constant Spike Source block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the inhibited neurons.
        :param sim.StaticSynapse conn: The connection to use. Internal inhibitory synapse by default, it is recommended NOT to use this parameter.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Inhibitory" by default, it is recommended NOT to use this parameter.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: Unused.
        :param list component_indexes: A list of indices used to select components from the list of components.
        :return: The number of connections that have been created.
        :rtype: int
        :raise ValueError: If this functions is used with the classic AND variant.
        """
        if conn is None:
            conn = self.inh_synapse

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_inhibition", input_population, self.and_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, component_indexes=None):
        """
        Connects an input population to the input neurons of the blocks.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the input neurons.
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

        created_connections = multiple_connect("connect_inputs", input_population, self.and_array, conn,
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

        created_connections = multiple_connect("connect_outputs", output_population, self.and_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_inhibited_neurons(self, flat=False):
        """
        Gets a list containing all the neurons inhibited by the Constant Spike Source block.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The list containing all the inhibited neurons of the blocks
        :rtype: list
        """
        inhibited_neurons = []

        for i in range(self.n_components):
            inhibited_neurons.append(self.and_array[i].get_inhibited_neuron())

        if flat:
            return flatten(inhibited_neurons)
        else:
            return inhibited_neurons

    def get_input_neurons(self, flat=False):
        """
        Gets a list containing all the input neurons of the blocks.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The flattened or unflattened list containing all the input neurons of the blocks
        :rtype: list
        """
        input_neurons = []

        for i in range(self.n_components):
            input_neurons.append(self.and_array[i].get_input_neurons())

        if flat:
            return flatten(input_neurons)
        else:
            return input_neurons

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the blocks.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The list containing all the output neurons of the blocks
        :rtype: list
        """
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.and_array[i].get_output_neurons())

        if flat:
            return flatten(output_neurons)
        else:
            return output_neurons
