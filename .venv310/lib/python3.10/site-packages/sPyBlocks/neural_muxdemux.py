from .connection_functions import truth_table_column, inverse_rcp_type, flatten
from .neural_and import MultipleNeuralAnd
from .neural_not import MultipleNeuralNot
from .neural_or import NeuralOr


class NeuralMuxDemux:
    def __init__(self, n_select, sim, global_params, neuron_params, std_conn, build_type="mux", and_type="classic"):
        # Storing parameters
        self.n_select = n_select
        self.n_outputs = 2 ** n_select
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        if build_type == "mux" or build_type == "demux":
            self.build_type = build_type
        else:
            raise ValueError("This build type is not implemented.")
        self.and_type = and_type

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        self.not_gates = MultipleNeuralNot(n_select, sim, global_params, neuron_params, std_conn)
        self.and_gates = MultipleNeuralAnd(self.n_outputs, n_select + 1, sim, global_params, neuron_params, std_conn,
                                           build_type=and_type)

        self.total_neurons += self.not_gates.total_neurons + self.and_gates.total_neurons
        self.total_internal_connections += self.not_gates.total_internal_connections + self.and_gates.total_internal_connections

        if build_type == "mux":
            self.or_gate = NeuralOr(sim, global_params, neuron_params, std_conn)

            self.total_neurons += self.or_gate.total_neurons
            self.total_internal_connections += self.or_gate.total_internal_connections

        # Create the connections
        created_connections = 0

        for i in range(n_select):
            created_connections += self.and_gates.connect_inputs(self.not_gates.not_array[i].output_neuron,
                                                                 component_indexes=truth_table_column(self.n_outputs, i,
                                                                                                      select=0))
        if build_type == "mux":
            created_connections += self.and_gates.connect_outputs(self.or_gate.output_neuron)

        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = self.not_gates.delay + self.std_conn.delay + self.and_gates.delay

    def connect_constant_spikes(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                                ini_pop_indexes=None, end_pop_indexes=None):

        if conn is None:
            conn = self.std_conn

        created_connections = 0

        # Excite NOT neurons
        created_connections += self.not_gates.connect_excitation(input_population, conn, conn_all=conn_all,
                                                                 rcp_type=rcp_type)

        # Inhibit AND neurons associated to each excited NOT neuron (Fast AND)
        if self.and_type == "fast":
            inv_rcp_type = inverse_rcp_type(rcp_type)
            created_connections += self.and_gates.connect_inhibition(input_population, conn=None, conn_all=conn_all,
                                                                     rcp_type=inv_rcp_type)

        self.total_input_connections += created_connections
        return created_connections

    def connect_signals(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                        ini_pop_indexes=None, end_pop_indexes=None, not_indexes=None):

        if conn is None:
            conn = self.std_conn

        if not_indexes is None:
            not_indexes = range(self.n_select)

        inv_rcp_type = inverse_rcp_type(rcp_type)

        created_connections = 0

        # Connect inputs to NOT neurons
        created_connections += self.not_gates.connect_inputs(input_population, conn, conn_all=conn_all,
                                                             rcp_type=inv_rcp_type, ini_pop_indexes=ini_pop_indexes,
                                                             component_indexes=not_indexes)

        # Connect inputs to AND neurons
        delayed_conn = self.sim.StaticSynapse(weight=conn.weight, delay=conn.delay * 2 + self.not_gates.delay)

        # TODO: Extract a new function?
        if ini_pop_indexes is None:  # All inputs for all specified components
            for i in range(len(not_indexes)):
                created_connections += self.and_gates.connect_inputs(input_population, delayed_conn,
                                                                     rcp_type=rcp_type,
                                                                     component_indexes=truth_table_column(
                                                                         self.n_outputs, not_indexes[i],
                                                                         select=1))
        else:
            if isinstance(ini_pop_indexes[0], list):  # Selected inputs for each specified component
                for i in range(len(not_indexes)):
                    created_connections += self.and_gates.connect_inputs(input_population, delayed_conn,
                                                                         rcp_type=rcp_type,
                                                                         ini_pop_indexes=ini_pop_indexes[i],
                                                                         component_indexes=truth_table_column(
                                                                             self.n_outputs,
                                                                             not_indexes[i],
                                                                             select=1))
            elif isinstance(ini_pop_indexes[0], int):  # Selected inputs for all specified components
                for i in range(len(not_indexes)):
                    created_connections += self.and_gates.connect_inputs(input_population, delayed_conn,
                                                                         rcp_type=rcp_type,
                                                                         ini_pop_indexes=ini_pop_indexes,
                                                                         component_indexes=truth_table_column(
                                                                             self.n_outputs,
                                                                             not_indexes[i],
                                                                             select=1))
            else:
                raise TypeError("pop_indexes must be an array of ints or an array of arrays of ints")

        self.total_input_connections += created_connections
        return created_connections

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, and_indexes=None):

        if conn is None:
            conn = self.std_conn

        if and_indexes is None:
            and_indexes = range(self.n_outputs)

        delayed_conn = self.sim.StaticSynapse(weight=conn.weight, delay=conn.delay * 2 + self.not_gates.delay)
        created_connections = self.and_gates.connect_inputs(input_population, delayed_conn, conn_all=conn_all,
                                                            rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                                            component_indexes=and_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, and_indexes=None):

        if conn is None:
            conn = self.std_conn

        if self.build_type == "mux":
            created_connections = self.or_gate.connect_outputs(output_population, conn, rcp_type=rcp_type,
                                                               end_pop_indexes=end_pop_indexes)

        else:
            if and_indexes is None:
                and_indexes = range(self.n_outputs)

            created_connections = self.and_gates.connect_outputs(output_population, conn, conn_all=conn_all,
                                                                 rcp_type=rcp_type, end_pop_indexes=end_pop_indexes,
                                                                 component_indexes=and_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_supplied_neurons(self):
        supplied_neurons = []

        for i in range(self.n_select):
            supplied_by_channel = [self.not_gates.not_array[i].get_excited_neuron()]
            for j in truth_table_column(self.n_outputs, i, select=1):
                supplied_by_channel.append(self.and_gates.and_array[j].get_inhibited_neuron())

            supplied_neurons.append(flatten(supplied_by_channel))

        return supplied_neurons

    def get_signal_neurons(self):
        signal_neurons = []

        for i in range(self.n_select):
            inputs_by_channel = [self.not_gates.not_array[i].get_input_neuron()]
            for j in truth_table_column(self.n_outputs, i, select=1):
                inputs_by_channel.append(self.and_gates.and_array[j].get_input_neurons())

            signal_neurons.append(flatten(inputs_by_channel))

        return signal_neurons

    def get_input_neurons(self):
        input_neurons = []

        for i in range(self.n_outputs):
            input_neurons.append(self.and_gates.and_array[i].get_input_neurons())

        return input_neurons

    def get_output_neurons(self):
        if self.build_type == "mux":
            return self.or_gate.output_neuron
        else:
            output_neurons = []

            for i in range(self.n_outputs):
                output_neurons.append(self.and_gates.and_array[i].get_output_neurons())

            return output_neurons


'''
# TODO: Add parameters to manage multiplexor connections
class MultipleNeuralMultiplexor:
    def __init__(self, n_components, n_inputs, sim, global_params, neuron_params, std_conn):
        # Storing parameters
        self.n_components = n_components
        self.n_inputs = n_inputs
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn

        # Create the array of multiple src
        self.multiplexor_array = []
        for i in range(n_components):
            self.multiplexor_array.append(NeuralMultiplexor(n_inputs, sim, global_params, neuron_params, std_conn))

        # Total output delay
        self.multiplexor_delay = self.multiplexor_array[0].multiplexor_delay

    def connect_signals(self, input_population, conn=None, rcp_type="excitatory", pop_indexes=None,
                        component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        multiple_connect("connect_signals", input_population, self.multiplexor_array, self.sim, conn, rcp_type=rcp_type,
                         pop_indexes=pop_indexes, component_indexes=component_indexes)

    def connect_inputs(self, input_population, conn=None, rcp_type="excitatory", pop_indexes=None,
                       component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        multiple_connect("connect_inputs", input_population, self.multiplexor_array, self.sim, conn, rcp_type=rcp_type,
                         pop_indexes=pop_indexes, component_indexes=component_indexes)

    def connect_outputs(self, output_population, conn=None, rcp_type="excitatory", pop_indexes=None,
                        component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        multiple_connect("connect_outputs", output_population, self.multiplexor_array, self.sim, conn, rcp_type=rcp_type,
                         pop_indexes=pop_indexes, component_indexes=component_indexes)
 '''
