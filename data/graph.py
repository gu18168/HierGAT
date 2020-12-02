def _format_graph(graph, id):
    graph = list(graph)
    graph.sort()
    graph.insert(0, id)
    return graph

def get_graph(entities, attribute, id):
    attr_start_index = len(entities)
    entity_graph = []
    attribute_graph = []

    # f(node) -> index
    # Example: f("Apple") = 20
    node_to_idx = {}

    for e_idx, entity in enumerate(entities):
        # use set to remove duplicate number
        entity_sub_graph = set()
        sentence = entity.to_attr(attribute)
        tokens = list(filter(lambda x: x, sentence.split(' ')))
        
        for token in tokens:
            # remove nan
            if token == 'nan':
                continue

            if token in node_to_idx:
                idx = node_to_idx.get(token)
                attribute_graph[idx - attr_start_index].add(e_idx)
                entity_sub_graph.add(idx)
            else:
                # Because node_to_idx contains the id of the entity
                # it should be subtracted (- e_idx)
                idx = attr_start_index + len(node_to_idx) - e_idx
                attribute_sub_graph = {e_idx}
                attribute_graph.append(attribute_sub_graph)
                entity_sub_graph.add(idx)

                node_to_idx[token] = idx

        # because set is un-order
        # so we convet it to list and add the id manully
        entity_sub_graph = _format_graph(entity_sub_graph, e_idx)
        entity_graph.append(entity_sub_graph)

        node_to_idx[getattr(entity, id)] = e_idx

    new_attribute_graph = []
    for index, attribute_sub_graph in enumerate(attribute_graph):
        attribute_sub_graph = _format_graph(
            attribute_sub_graph, index + attr_start_index)
        new_attribute_graph.append(attribute_sub_graph)

    graph = entity_graph + new_attribute_graph
    return graph, node_to_idx


def get_line(entities, node_to_idx, attribute):
    line = []
    for entity in entities:
        tokens = list(filter(lambda x: x and x != 'nan', entity.to_attr(attribute).split(' ')))
        line.append(list(map(lambda x: node_to_idx[x], tokens)))

    return line