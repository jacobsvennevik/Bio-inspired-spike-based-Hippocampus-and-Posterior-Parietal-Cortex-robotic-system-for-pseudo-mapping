from math import ceil, log2

from sPyBlocks.connection_functions import flatten, inverse_rcp_type
from sPyBlocks.neural_decoder import NeuralDecoder
from sPyBlocks.neural_latch_d import MultipleNeuralLatchD
from sPyBlocks.neural_not import MultipleNeuralNot


class NeuralMemory:
    def __init__(self, n_dir, width, sim, global_params, neuron_params, std_conn, and_type="classic"):
        # Storing parameters
        self.n_dir = n_dir
        self.width = width
        self.capacity = n_dir * width
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
        self.decoder = NeuralDecoder(ceil(log2(n_dir + 1)), sim, global_params, neuron_params, std_conn,
                                     and_type=and_type)
        self.not_gates = MultipleNeuralNot(width, sim, global_params, neuron_params, std_conn)

        self.latches = MultipleNeuralLatchD(self.capacity, sim, global_params, neuron_params, std_conn,
                                            and_type=and_type, include_not=False)

        self.total_neurons += self.decoder.total_neurons + self.latches.total_neurons
        self.total_internal_connections += self.decoder.total_internal_connections + self.latches.total_internal_connections

        # Create the connections
        created_connections = 0

        for bit in range(0, width):
            created_connections += self.latches.connect_not_data(self.not_gates.not_array[bit].output_neuron,
                                                                 component_indexes=range(bit, self.capacity, width))

        for i in range(1, n_dir + 1):  # Output 0 from decoder is not connected
            created_connections += self.latches.connect_signals(self.decoder.and_gates.and_array[i].output_neuron,
                                                                component_indexes=range((i - 1) * width, i * width))

        self.total_internal_connections += created_connections

        # Total internal delay
        self.write_delay = self.decoder.delay + self.latches.delay

    def connect_constant_spikes(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                                ini_pop_indexes=None, end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = self.decoder.connect_constant_spikes(input_population, conn, conn_all=conn_all,
                                                                   rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

        created_connections += self.not_gates.connect_excitation(input_population, conn, conn_all=conn_all,
                                                                 rcp_type=rcp_type,
                                                                 ini_pop_indexes=ini_pop_indexes)

        created_connections += self.latches.connect_constant_spikes(input_population, conn, conn_all=conn_all,
                                                                    rcp_type=rcp_type,
                                                                    ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_data(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                     ini_pop_indexes=None, end_pop_indexes=None, bit_indexes=None):
        if conn is None:
            conn = self.std_conn

        if bit_indexes is None:
            bit_indexes = range(self.width)

        inv_rcp_type = inverse_rcp_type(rcp_type)
        delayed_conn_data = self.sim.StaticSynapse(weight=conn.weight,
                                                   delay=self.std_conn.delay + self.decoder.delay + conn.delay)
        delayed_conn_not_data = self.sim.StaticSynapse(weight=conn.weight,
                                                       delay=self.std_conn.delay + self.decoder.delay)

        created_connections = 0

        # TODO: Extract a new function?
        if ini_pop_indexes is None:  # All inputs for all specified components
            for i in range(len(bit_indexes)):
                created_connections += self.latches.connect_data(input_population, delayed_conn_data, conn_all=conn_all,
                                                                 rcp_type=rcp_type,
                                                                 component_indexes=range(bit_indexes[i],
                                                                                         self.capacity,
                                                                                         self.width))

            created_connections += self.not_gates.connect_inputs(input_population, delayed_conn_not_data,
                                                                 conn_all=conn_all,
                                                                 rcp_type=inv_rcp_type,
                                                                 component_indexes=bit_indexes)
        else:
            if isinstance(ini_pop_indexes[0], list):  # Selected inputs for each specified component
                for i in range(len(bit_indexes)):
                    created_connections += self.latches.connect_data(input_population, delayed_conn_data,
                                                                     conn_all=conn_all,
                                                                     rcp_type=rcp_type,
                                                                     ini_pop_indexes=ini_pop_indexes[i],
                                                                     component_indexes=range(bit_indexes[i],
                                                                                             self.capacity,
                                                                                             self.width))

                    created_connections += self.not_gates.connect_inputs(input_population, delayed_conn_not_data,
                                                                         conn_all=conn_all,
                                                                         rcp_type=inv_rcp_type,
                                                                         ini_pop_indexes=ini_pop_indexes[i],
                                                                         component_indexes=[i])
            elif isinstance(ini_pop_indexes[0], int):  # Selected inputs for all specified components
                for i in range(len(bit_indexes)):
                    created_connections += self.latches.connect_data(input_population, delayed_conn_data,
                                                                     conn_all=conn_all,
                                                                     rcp_type=rcp_type,
                                                                     ini_pop_indexes=ini_pop_indexes,
                                                                     component_indexes=range(bit_indexes[i],
                                                                                             self.capacity,
                                                                                             self.width))

                created_connections += self.not_gates.connect_inputs(input_population, delayed_conn_not_data,
                                                                     conn_all=conn_all,
                                                                     rcp_type=inv_rcp_type,
                                                                     ini_pop_indexes=ini_pop_indexes,
                                                                     component_indexes=bit_indexes)
            else:
                raise TypeError("pop_indexes must be an array of ints or an array of arrays of ints")

        self.total_input_connections += created_connections
        return created_connections

    def connect_signals(self, input_population, conn=None, conn_all=True, rcp_type="excitatory",
                        ini_pop_indexes=None, end_pop_indexes=None, not_indexes=None):
        if conn is None:
            conn = self.std_conn

        if not_indexes is None:
            not_indexes = range(self.decoder.n_inputs)

        created_connections = self.decoder.connect_inputs(input_population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                          ini_pop_indexes=ini_pop_indexes, not_indexes=not_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, ff_indexes=None):
        if conn is None:
            conn = self.std_conn

        if ff_indexes is None:
            ff_indexes = range(self.capacity)

        created_connections = self.latches.connect_outputs(output_population, conn, conn_all=conn_all,
                                                           rcp_type=rcp_type, end_pop_indexes=end_pop_indexes,
                                                           component_indexes=ff_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_supplied_neurons(self, flat=False):
        supplied_neurons = [self.decoder.get_supplied_neurons(), self.not_gates.get_excited_neurons(),
                            self.latches.get_supplied_neurons()]

        if flat:
            return flatten(supplied_neurons)
        else:
            return supplied_neurons

    def get_data_neurons(self, flat=False):
        data_neurons = self.latches.get_data_neurons()

        if flat:
            return flatten(data_neurons)
        else:
            return data_neurons

    def get_signal_neurons(self, flat=False):
        signal_neurons = self.decoder.get_input_neurons()

        if flat:
            return flatten(signal_neurons)
        else:
            return signal_neurons

    def get_output_neurons(self, flat=False):
        # TODO: Modify this when AND gates are introduced to allow reading
        output_neurons = self.latches.get_output_neurons()

        if flat:
            return flatten(output_neurons)
        else:
            return output_neurons
