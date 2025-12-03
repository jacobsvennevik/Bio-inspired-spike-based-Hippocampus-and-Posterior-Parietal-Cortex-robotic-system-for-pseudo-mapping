from .connection_functions import inverse_rcp_type, flatten
from .neural_and import MultipleNeuralAnd
from .neural_not import NeuralNot


class NeuralFlankDetector:
    """
    This class defines the flank detector block, which allows to detect rising and falling edges.
    """
    def __init__(self, sim, global_params, neuron_params, std_conn, and_type="classic"):
        """
        Constructor of the class.

        :param sim: The simulator package.
        :param dict global_params: A dictionary of type str:int which must include the "min_delay" keyword. This keyword is likely to have the time period associated with it as a value.
        :param dict neuron_params: A dictionary of type str:int containing the neuron parameters.
        :param sim.StaticSynapse std_conn: The connection to be used for the construction of the block. Commonly, its weight is 1.0 and its delay is equal to the timestep.
        :param str and_type: A string indicating the AND variant ("classic" or "fast"). "classic" by default.
        """
        # Storing parameters
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        self.and_type = and_type

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        self.not_gate = NeuralNot(sim, global_params, neuron_params, std_conn)
        self.and_gates = MultipleNeuralAnd(2, 2, sim, global_params, neuron_params, std_conn, build_type=and_type)

        self.total_neurons += self.not_gate.total_neurons + self.and_gates.total_neurons
        self.total_internal_connections += self.not_gate.total_internal_connections + self.and_gates.total_internal_connections

        # Create the connections
        created_connections = 0
        created_connections += self.and_gates.and_array[0].connect_inputs(self.not_gate.output_neuron)  # Rising output
        created_connections += self.and_gates.and_array[1].connect_inputs(self.not_gate.output_neuron)  # Falling output
        self.total_internal_connections += created_connections

        # Total internal delay
        self.rising_delay = std_conn.delay + self.and_gates.delay  # Rising path (shortest path)
        self.falling_delay = self.not_gate.delay + std_conn.delay * 2 + self.and_gates.delay  # NOT path (shortest path)

    def connect_constant_spikes(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                                ini_pop_indexes=None, end_pop_indexes=None):
        """
        Connects an input population to the neurons to be inhibited or excited by the Constant Spike Source block.

        :param sim.Population, sim.PopulationView, sim.Assembly, list input_population: A PyNN object or a list of PyNN objects containing the population to connect to the inhibited or excited neurons.
        :param sim.StaticSynapse conn: The connection to use.
        :param conn_all: Unused.
        :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory). "Excitatory" by default, it is recommended NOT to use this parameter.
        :param list ini_pop_indexes: A list of indices used to select objects from the input population.
        :param end_pop_indexes: Unused.
        :return: The number of connections that have been created.
        :rtype: int
        """
        if conn is None:
            conn = self.std_conn

        created_connections = self.not_gate.connect_excitation(input_population, conn, rcp_type=rcp_type,
                                                               ini_pop_indexes=ini_pop_indexes)

        if self.and_type == "fast":
            inv_rcp_type = inverse_rcp_type(rcp_type)
            created_connections += self.and_gates.connect_inhibition(input_population, conn, rcp_type=inv_rcp_type,
                                                                     ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

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

        inv_rcp_type = inverse_rcp_type(rcp_type)
        created_connections += self.not_gate.connect_inputs(input_population, conn, rcp_type=inv_rcp_type,
                                                            ini_pop_indexes=ini_pop_indexes)

        rising_conn = self.sim.StaticSynapse(weight=conn.weight, delay=conn.delay + self.not_gate.delay)
        created_connections += self.and_gates.connect_inputs(input_population, rising_conn, rcp_type=rcp_type,
                                                             ini_pop_indexes=ini_pop_indexes,
                                                             component_indexes=[0])

        falling_conn = self.sim.StaticSynapse(weight=conn.weight, delay=conn.delay * 3 + self.not_gate.delay)
        created_connections += self.and_gates.connect_inputs(input_population, falling_conn, rcp_type=rcp_type,
                                                             ini_pop_indexes=ini_pop_indexes,
                                                             component_indexes=[1])

        self.total_input_connections += created_connections
        return created_connections

    def connect_rising_edge(self, output_population, conn=None, conn_all=True, rcp_type="excitatory",
                            ini_pop_indexes=None, end_pop_indexes=None):
        """
        Connects the output AND of the block associated with the rising edge to an output population.

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

        created_connections = self.and_gates.connect_outputs(output_population, conn, rcp_type=rcp_type,
                                                             end_pop_indexes=end_pop_indexes,
                                                             component_indexes=[0])

        self.total_output_connections += created_connections
        return created_connections

    def connect_falling_edge(self, output_population, conn=None, conn_all=True, rcp_type="excitatory",
                             ini_pop_indexes=None, end_pop_indexes=None):
        """
        Connects the output AND of the block associated with the falling edge to an output population.

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

        created_connections = self.and_gates.connect_outputs(output_population, conn, rcp_type=rcp_type,
                                                             end_pop_indexes=end_pop_indexes,
                                                             component_indexes=[1])

        self.total_output_connections += created_connections
        return created_connections

    def get_supplied_neurons(self, flat=False):
        """
        Gets a list containing all the neurons inhibited or excited by the Constant Spike Source block.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The list containing all the inhibited neurons of the block
        :rtype: list
        """
        if self.and_type == "classic":
            return self.not_gate.get_output_neuron()
        else:
            if flat:
                return flatten([self.not_gate.get_output_neuron(), self.and_gates.get_output_neurons()])
            else:
                return [self.not_gate.get_output_neuron(), self.and_gates.get_output_neurons()]

    def get_input_neurons(self, flat=False):
        """
        Gets a list containing all the input neurons of the block.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The flattened or unflattened list containing all the input neurons of the block
        :rtype: list
        """
        if flat:
            return flatten([self.not_gate.get_input_neuron(), self.and_gates.get_input_neurons()])
        else:
            return [self.not_gate.get_input_neuron(), self.and_gates.get_input_neurons()]

    def get_output_neurons(self, flat=False):
        """
        Gets a list containing all the output neurons of the block.

        :param bool flat: A boolean value indicating whether or not to flatten the list. False by default.
        :return: The list containing all the output neurons of the block
        :rtype: list
        """
        if flat:
            return flatten(self.and_gates.get_output_neurons())
        else:
            return self.and_gates.get_output_neurons()
