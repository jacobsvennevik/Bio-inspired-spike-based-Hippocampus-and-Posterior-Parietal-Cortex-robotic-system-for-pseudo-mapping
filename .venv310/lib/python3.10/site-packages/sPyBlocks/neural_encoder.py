from math import log2

from sPyBlocks.neural_or import MultipleNeuralOr


class NeuralEncoder:
    def __init__(self, n_inputs, sim, global_params, neuron_params, std_conn):
        # Storing parameters
        self.n_inputs = n_inputs
        self.n_outputs = int(log2(n_inputs))
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
        self.or_gates = MultipleNeuralOr(self.n_outputs, sim, global_params, neuron_params, std_conn)

        self.total_neurons += self.or_gates.total_neurons
        self.total_internal_connections += self.or_gates.total_internal_connections

        # Total internal delay
        self.delay = self.or_gates.delay

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                       ini_pop_indexes=None, end_pop_indexes=None, or_indexes=None):

        if conn is None:
            conn = self.std_conn

        if or_indexes is None:
            or_indexes = range(self.n_outputs)

        created_connections = self.or_gates.connect_inputs(input_population, conn, rcp_type=rcp_type,
                                                           ini_pop_indexes=ini_pop_indexes,
                                                           component_indexes=or_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, or_indexes=None):

        if conn is None:
            conn = self.std_conn

        if or_indexes is None:
            or_indexes = range(self.n_outputs)

        created_connections = self.or_gates.connect_outputs(output_population, conn, conn_all=conn_all,
                                                            rcp_type=rcp_type, end_pop_indexes=end_pop_indexes,
                                                            component_indexes=or_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neurons(self):
        return self.or_gates.get_input_neurons()

    def get_output_neurons(self):
        return self.or_gates.get_output_neurons()