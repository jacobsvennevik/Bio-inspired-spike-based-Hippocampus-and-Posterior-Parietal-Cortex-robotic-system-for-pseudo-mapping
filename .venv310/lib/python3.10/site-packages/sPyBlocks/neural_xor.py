import numpy as np

from sPyBlocks.connection_functions import create_connections, multiple_connect, flatten


class NeuralXor:
    """
    This class defines the XOR block.
    """
    def __init__(self, n_inputs, sim, global_params, neuron_params, std_conn):
        """
        Constructor of the class.

        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the block. Commonly, its weight is 1.0 and its delay is equal to the timestep.
        """
        # Storing parameters
        self.n_inputs = n_inputs
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
        self.input_neurons = sim.Population(n_inputs, sim.IF_curr_exp(**neuron_params),
                                            initial_values={'v': neuron_params["v_rest"]})
        self.x_neurons = sim.Population(n_inputs, sim.IF_curr_exp(**neuron_params),
                                        initial_values={'v': neuron_params["v_rest"]})

        self.total_neurons += self.input_neurons.size + self.x_neurons.size

        # Create the connections
        created_connections = 0

        created_connections += create_connections(self.input_neurons, self.x_neurons, sim, std_conn, False)

        input_indexes = range(n_inputs)
        for i in input_indexes:
            inh_indexes = np.delete(input_indexes, i)

            created_connections += create_connections(self.input_neurons, self.x_neurons, sim, std_conn,
                                                      rcp_type="inhibitory", ini_pop_indexes=[i],
                                                      end_pop_indexes=inh_indexes)

        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = self.std_conn.delay

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        """
        Connects an input population to the input neurons of the block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the input neurons.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: A boolean indicating whether or not all selected input objects should be connected to all selected input neurons of the block.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: A list of indices used to select objects from the set of input neurons of the block.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(input_population, self.input_neurons, self.sim, conn,
                                                 conn_all=conn_all,
                                                 rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                                 end_pop_indexes=end_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None):
        """
        Connects the output neurons of the block to an output population.

        :param sim.Population, sim.PopulationView, sim.Assembly, list output_population: A PyNN object or a list of PyNN objects containing the population to connect the output neurons to.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: A boolean indicating whether or not all selected output neurons of the block should be connected to all selected output neurons.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: A list of indices used to select objects from the set of output neurons of the block.
        :param end_pop_indexes: A list of indices used to select objects from the output population.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(self.x_neurons, output_population, self.sim, conn, conn_all=conn_all,
                                                 rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                                 end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neurons(self, flat=False):
        """
        Gets a list containing all the input neurons of the block.

        :param bool flat: Unused.
        :return: The flattened or unflattened list containing all the input neurons of the block
        :rtype: list
        """
        return [self.input_neurons]

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the block.

        :param bool flat: Unused.
        :return: The list containing all the output neurons of the block
        :rtype: list
        """
        return [self.x_neurons]


class MultipleNeuralXor:
    """
    This class allows to create multiple XOR blocks of the same type.
    """
    def __init__(self, n_components, n_inputs, sim, global_params, neuron_params, std_conn):
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
        self.n_inputs = n_inputs
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
        self.xor_array = []
        for i in range(n_components):
            xor_gate = NeuralXor(n_inputs, sim, global_params, neuron_params, std_conn)
            self.xor_array.append(xor_gate)

            self.total_neurons += xor_gate.total_neurons
            self.total_internal_connections += xor_gate.total_internal_connections

        # Total internal delay
        self.delay = self.xor_array[0].delay

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, component_indexes=None):
        """
        Connects an input population to the input neurons of the block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the input neurons.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: A boolean indicating whether or not all selected input objects should be connected to all selected input neurons of the block.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: A list of indices used to select objects from the set of input neurons of the block.
        :param list component_indexes: A list of indices used to select components from the list of components.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_inputs", input_population, self.xor_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, component_indexes=None):
        """
        Connects the output neurons of the block to an output population.

        :param sim.Population, sim.PopulationView, sim.Assembly, list output_population: A PyNN object or a list of PyNN objects containing the population to connect the output neurons to.
        :param sim.StaticSynapse conn: The connection to use. std_conn (class parameter) by default.
        :param conn_all: A boolean indicating whether or not all selected output neurons of the block should be connected to all selected output neurons.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default.
        :param list ini_pop_indexes: A list of indices used to select objects from the set of output neurons of the block.
        :param end_pop_indexes: A list of indices used to select objects from the output population.
        :param list component_indexes: A list of indices used to select components from the list of components.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_outputs", output_population, self.xor_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neurons(self, flat=False):
        """
        Gets a list containing all the input neurons of the block.

        :param bool flat: Unused.
        :return: The flattened or unflattened list containing all the input neurons of the block
        :rtype: list
        """
        input_neurons = []

        for i in range(self.n_components):
            input_neurons.append(self.xor_array[i].get_input_neurons())

        if flat:
            return flatten(input_neurons)
        else:
            return [input_neurons]

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the block.

        :param bool flat: Unused.
        :return: The list containing all the output neurons of the block
        :rtype: list
        """
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.xor_array[i].get_output_neurons())

        if flat:
            return flatten(output_neurons)
        else:
            return [output_neurons]
