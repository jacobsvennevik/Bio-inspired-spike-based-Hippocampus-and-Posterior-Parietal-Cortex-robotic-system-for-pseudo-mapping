import numpy as np


def truth_table_column(n_values, n_var, select=1):
    """
    Generates the array of indices where the value indicated by the "select" parameter is found in the column
    indicated by the "n_var" parameter. "n_values" indicates the number of elements to consider in that column,
    commonly 2^n if n is the number of variables in the truth table (it is the number of all possible binary
    combinations of the input variables).

    :param int n_values: The number of elements of the column to consider.
    :param int n_var: The index of the input variable or column. From 0 to n-1 if n is the number of variables in the truth table. Column 0 is the least significant column.
    :param int select: Value to consider in the truth table column. There are only two possibilities: 0 or 1, since only binary values can be entered in a truth table.
    :return: If select = 0, it returns a list (zeros) containing the indexes of 0 values in the selected column. Otherwise, it returns a list (ones) containing the indexes of 1 values in the selected column.
    :rtype: list
    :raise ValueError: If select is not 0 or 1.
    """

    if select != 1 and select != 0:
        raise ValueError("Only binary values are allowed in the truth table")

    numbers = range(0, n_values)

    zeros = []
    for i in numbers:
        if i % (2 ** (n_var + 1)) == 0:
            for j in range(i, i + 2 ** n_var):
                if i >= n_values:
                    break
                zeros.append(i)
                i += 1

    if not select:
        return zeros
    else:
        ones = np.delete(numbers, zeros)
        return ones.tolist()


def is_pynn_object(obj, sim):
    """
    Checks if the object passed as a parameter belongs to a type of the PyNN library.

    :param obj: The object to check.
    :param sim: The simulator package containing the defined classes.
    :return: True if the object is an instance of a PyNN class. Otherwise, false.
    :rtype: bool
    """
    is_population = isinstance(obj, sim.Population)
    is_population_view = isinstance(obj, sim.PopulationView)
    is_assembly = isinstance(obj, sim.Assembly)

    return is_population or is_population_view or is_assembly


def inverse_rcp_type(rcp_type):
    """
    Generates a string indicating the receiver type inverse to the one received in the input variable. Only the
    "excitatory" or "inhibitory" strings are allowed; any other string will throw an exception.

    :param str rcp_type: The original receptor type.
    :return: "Inhibitory" if rcp_type is "excitatory". Otherwise, "excitatory".
    :rtype: str
    :raise ValueError: If rcp_type is not "excitatory" or "inhibitory".
    """
    if rcp_type == "excitatory":
        return "inhibitory"
    elif rcp_type == "inhibitory":
        return "excitatory"
    else:
        raise ValueError("This receptor type is not supported")


def list_element(array, index_array):
    """
    Returns the elements of the input list found at the positions indicated by the indices contained in index_array[
    0]. Accessing the first position of index_array allows to optimize the create_connections function, using a notation
    similar to that needed to use PyNN's PopulationView function.

    :param list array: A list of generic elements.
    :param list index_array: A list containing the positions of the elements to take in the input list.
    :return: A list containing the desired elements.
    :rtype: list
    """
    return array[index_array[0]]


def create_connections(ini_pop, end_pop, sim, conn, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
    """
    Creates connections between ini_pop and end_pop objects.

    :param sim.Population, sim.PopulationView, sim.Assembly, list ini_pop: A PyNN object or a list of PyNN objects that serve as input population. Starting point of the connections.
    :param sim.Population, sim.PopulationView, sim.Assembly, list end_pop: A PyNN object or a list of PyNN objects that serve as end population. End point of the connections.
    :param sim: The simulator package.
    :param sim.StaticSynapse conn: The connection to use.
    :param bool conn_all: A boolean indicating whether or not all selected input objects should be connected to all selected output objects.
    :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory).
    :param list ini_pop_indexes: A list of indices used to select objects from the input population.
    :param list end_pop_indexes: A list of indices used to select objects from the output population.
    :return: The number of connections that have been created.
    :rtype: int
    """
    # Check ini_pop and end_pop object types
    ini_pop_islist = isinstance(ini_pop, list)
    end_pop_islist = isinstance(end_pop, list)

    # Functions to access elements of ini_pop and end_pop and calculation of ini_pop and end_pop sizes
    if not ini_pop_islist:
        ini_pop_function = sim.PopulationView
        ini_pop_size = ini_pop.size
    else:
        ini_pop_function = list_element
        ini_pop_size = len(ini_pop)

    if not end_pop_islist:
        end_pop_function = sim.PopulationView
        end_pop_size = end_pop.size
    else:
        end_pop_function = list_element
        end_pop_size = len(end_pop)

    # If ini_pop_indexes or end_pop_indexes are None, take all elements in ini_pop or end_pop respectively
    if ini_pop_indexes is None:
        ini_pop_indexes = range(ini_pop_size)

    if end_pop_indexes is None:
        end_pop_indexes = range(end_pop_size)

    # Length of ini_pop_indexes and end_pop_indexes arrays
    ini_pop_indexes_len = len(ini_pop_indexes)
    end_pop_indexes_len = len(end_pop_indexes)

    # Create connections
    if conn_all:  # AllToAll
        for i in ini_pop_indexes:
            for j in end_pop_indexes:
                sim.Projection(ini_pop_function(ini_pop, [i]), end_pop_function(end_pop, [j]),
                               sim.OneToOneConnector(), conn, receptor_type=rcp_type)
        created_connections = ini_pop_indexes_len * end_pop_indexes_len
    else:  # OneToOne
        if ini_pop_indexes_len != end_pop_indexes_len:
            raise ValueError("The number of selected elements of ini_pop and end_pop must be the same in OneToOne connections")

        for i in range(ini_pop_indexes_len):  # It could be the length of end_pop_indexes too
            sim.Projection(ini_pop_function(ini_pop, [ini_pop_indexes[i]]),
                           end_pop_function(end_pop, [end_pop_indexes[i]]),
                           sim.OneToOneConnector(), conn, receptor_type=rcp_type)
        created_connections = ini_pop_indexes_len

    return created_connections


def multiple_connect(function_name, population, components, conn, conn_all, rcp_type, ini_pop_indexes,
                     end_pop_indexes, component_indexes):
    """
    Calls the function indicated by function_name for each component in components passing a population as
    input parameter.

    :param str function_name: The name of the function to call.
    :param sim.Population, sim.PopulationView, sim.Assembly, list population: A PyNN object or a list of PyNN objects containing the population to be passed as input parameter.
    :param list components: A list of spiking functional blocks defined by multiple type classes from this library.
    :param sim.StaticSynapse conn: The connection to use.
    :param bool conn_all: A boolean indicating whether or not all selected input objects should be connected to all selected output objects.
    :param str rcp_type: A string indicating the receptor type of the connections (excitatory or inhibitory).
    :param list ini_pop_indexes: A list of indices used to select objects from the input population.
    :param list end_pop_indexes: A list of indices used to select objects from the output population.
    :param list component_indexes: A list of indices used to select objects from the component list.
    :return: None
    """

    created_connections = 0

    # ini_pop_indexes is an array of arrays
    if ini_pop_indexes is not None and ini_pop_indexes != [] and isinstance(ini_pop_indexes[0], list):
        # end_pop_indexes is an array of arrays
        if end_pop_indexes is not None and end_pop_indexes != [] and isinstance(end_pop_indexes[0], list):
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes[i],
                                                        end_pop_indexes=end_pop_indexes[i])
        # end_pop_indexes is None or an array
        else:
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes[i],
                                                        end_pop_indexes=end_pop_indexes)
    # ini_pop_indexes is None or an array
    else:
        # end_pop_indexes is an array of arrays
        if end_pop_indexes is not None and end_pop_indexes != [] and isinstance(end_pop_indexes[0], list):
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes,
                                                        end_pop_indexes=end_pop_indexes[i])
        # end_pop_indexes is None or an array
        else:
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes,
                                                        end_pop_indexes=end_pop_indexes)

    return created_connections


def flatten(array):
    """
    Flats an array recursively.

    :param list array: An input array.
    :return: The flattened input array.
    :rtype: list
    """
    if not array:
        return array
    if isinstance(array[0], list):
        return flatten(array[0]) + flatten(array[1:])
    return array[:1] + flatten(array[1:])

