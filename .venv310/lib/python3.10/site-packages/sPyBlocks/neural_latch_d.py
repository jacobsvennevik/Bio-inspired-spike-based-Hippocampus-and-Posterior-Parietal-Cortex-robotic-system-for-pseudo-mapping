from sPyBlocks.connection_functions import inverse_rcp_type, flatten, multiple_connect
from sPyBlocks.neural_and import MultipleNeuralAnd
from sPyBlocks.neural_latch_sr import NeuralLatchSR
from sPyBlocks.neural_not import NeuralNot


class NeuralLatchD:
    def __init__(self, sim, global_params, neuron_params, std_conn, and_type="classic", include_not=True):
        # Storing parameters
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        self.and_type = and_type
        self.include_not = include_not

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        if include_not:
            self.not_gate = NeuralNot(sim, global_params, neuron_params, std_conn)
        self.and_gates = MultipleNeuralAnd(2, 2, sim, global_params, neuron_params, std_conn, build_type=and_type)
        self.latch_sr = NeuralLatchSR(sim, global_params, neuron_params, std_conn)

        self.total_neurons += self.and_gates.total_neurons + self.latch_sr.total_neurons
        self.total_internal_connections += self.and_gates.total_internal_connections + self.latch_sr.total_internal_connections
        if include_not:
            self.total_neurons += self.not_gate.total_neurons
            self.total_internal_connections += self.not_gate.total_internal_connections

        # Create the connections
        created_connections = 0
        if include_not:
            created_connections += self.and_gates.connect_inputs(self.not_gate.output_neuron, std_conn,
                                                                 component_indexes=[1])
        created_connections += self.latch_sr.connect_set(self.and_gates.and_array[0].output_neuron)
        created_connections += self.latch_sr.connect_reset(self.and_gates.and_array[1].output_neuron)
        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = std_conn.delay * 2 + self.and_gates.delay + self.latch_sr.delay
        if include_not:
            self.delay += std_conn.delay + self.not_gate.delay

    def connect_constant_spikes(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                                ini_pop_indexes=None, end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = 0

        if self.include_not:
            created_connections += self.not_gate.connect_excitation(input_population, conn, conn_all=conn_all,
                                                                    rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

        if self.and_type == "fast":
            inv_rcp_type = inverse_rcp_type(rcp_type)
            created_connections += self.and_gates.connect_inhibition(input_population, conn, conn_all=conn_all,
                                                                     rcp_type=inv_rcp_type,
                                                                     ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_data(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                     ini_pop_indexes=None, end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        if self.include_not:
            inv_rcp_type = inverse_rcp_type(rcp_type)
            created_connections = self.not_gate.connect_inputs(input_population, conn, conn_all=conn_all,
                                                               rcp_type=inv_rcp_type, ini_pop_indexes=ini_pop_indexes)

            delayed_conn = self.sim.StaticSynapse(weight=conn.weight,
                                                  delay=conn.delay + self.std_conn.delay + self.not_gate.delay)
            created_connections += self.and_gates.connect_inputs(input_population, delayed_conn, conn_all=conn_all,
                                                                 rcp_type=rcp_type,
                                                                 ini_pop_indexes=ini_pop_indexes, component_indexes=[0])
        else:
            created_connections = self.and_gates.connect_inputs(input_population, conn, conn_all=conn_all,
                                                                rcp_type=rcp_type,
                                                                ini_pop_indexes=ini_pop_indexes, component_indexes=[0])

        self.total_input_connections += created_connections
        return created_connections

    def connect_not_data(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                         ini_pop_indexes=None, end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = self.and_gates.connect_inputs(input_population, conn, conn_all=conn_all,
                                                            rcp_type=rcp_type,
                                                            ini_pop_indexes=ini_pop_indexes, component_indexes=[1])

        self.total_input_connections += created_connections
        return created_connections

    def connect_signal(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                       ini_pop_indexes=None, end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        if self.include_not:
            delayed_conn = self.sim.StaticSynapse(weight=conn.weight,
                                                  delay=conn.delay + self.std_conn.delay + self.not_gate.delay)
            created_connections = self.and_gates.connect_inputs(input_population, delayed_conn, conn_all=conn_all,
                                                                rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)
        else:
            created_connections = self.and_gates.connect_inputs(input_population, conn, conn_all=conn_all,
                                                                rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_output(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = self.latch_sr.connect_output(output_population, conn, conn_all=conn_all,
                                                           rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_supplied_neurons(self, flat=False):
        if flat:
            if self.include_not:
                return flatten([self.not_gate.get_excited_neuron(), self.and_gates.get_inhibited_neurons()])
            else:
                return flatten([self.and_gates.get_inhibited_neurons()])
        else:
            if self.include_not:
                return [self.not_gate.get_excited_neuron(), self.and_gates.get_inhibited_neurons()]
            else:
                return [self.and_gates.get_inhibited_neurons()]

    def get_data_neurons(self, flat=False):
        if flat:
            if self.include_not:
                return flatten([self.not_gate.get_input_neuron(), self.and_gates.and_array[0].get_input_neurons()])
            else:
                return flatten([self.and_gates.and_array[0].get_input_neurons()])
        else:
            if self.include_not:
                return [self.not_gate.get_input_neuron(), self.and_gates.and_array[0].get_input_neurons()]
            else:
                return [self.and_gates.and_array[0].get_input_neurons()]

    def get_signal_neurons(self, flat=False):
        return self.and_gates.get_input_neurons(flat=flat)

    def get_output_neuron(self):
        return self.latch_sr.get_output_neurons()


class MultipleNeuralLatchD:
    def __init__(self, n_components, sim, global_params, neuron_params, std_conn, and_type="classic", include_not=True):
        # Storing parameters
        self.n_components = n_components
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        self.and_type = and_type
        self.include_not = include_not

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the array of multiple src
        self.latch_array = []
        for i in range(n_components):
            latch = NeuralLatchD(sim, global_params, neuron_params, std_conn, and_type=and_type,
                                 include_not=include_not)
            self.latch_array.append(latch)

            self.total_neurons += latch.total_neurons
            self.total_internal_connections += latch.total_internal_connections

        # Total internal delay
        self.delay = self.latch_array[0].delay

    def connect_constant_spikes(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                                ini_pop_indexes=None, end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_constant_spikes", input_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_data(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                     ini_pop_indexes=None, end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_data", input_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_not_data(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                         ini_pop_indexes=None, end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_not_data", input_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_signals(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                        ini_pop_indexes=None, end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_signal", input_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_outputs", output_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_supplied_neurons(self, flat=False):
        supplied_neurons = []

        for i in range(self.n_components):
            supplied_neurons.append(self.latch_array[i].get_supplied_neurons())

        if flat:
            return flatten(supplied_neurons)
        else:
            return supplied_neurons

    def get_data_neurons(self, flat=False):
        data_neurons = []

        for i in range(self.n_components):
            data_neurons.append(self.latch_array[i].get_data_neurons())

        if flat:
            return flatten(data_neurons)
        else:
            return data_neurons

    def get_signal_neurons(self, flat=False):
        signal_neurons = []

        for i in range(self.n_components):
            signal_neurons.append(self.latch_array[i].get_signal_neurons())

        if flat:
            return flatten(signal_neurons)
        else:
            return signal_neurons

    def get_output_neurons(self, flat=False):
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.latch_array[i].get_output_neuron())

        if flat:
            return flatten(output_neurons)
        else:
            return output_neurons
