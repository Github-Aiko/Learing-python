from collections import defaultdict

def tarjan_scc(graph):
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    result = []
    on_stack = set()

    def strongconnect(node):
        # Set the depth index for this node to the smallest unused index
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        # Consider successors of this node
        for successor in graph[node]:
            if successor not in index:
                # Successor has not yet been visited; recurse on it
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in on_stack:
                # The successor is on the stack and hence in the current SCC
                lowlink[node] = min(lowlink[node], index[successor])

        # If this node is a root node, pop the stack and generate an SCC
        if lowlink[node] == index[node]:
            scc = []
            while True:
                successor = stack.pop()
                on_stack.remove(successor)
                scc.append(successor)
                if successor == node:
                    break
            result.append(scc)

    for node in graph:
        if node not in index:
            strongconnect(node)

    return result

def count_critical_wires(n, m, k, l, a, b, edges):
    # Create graph
    graph = defaultdict(list)
    for p, q in edges:
        graph[p].append(q)
        graph[q].append(p)

    # Separate nodes by service type
    a_nodes = set(a)
    b_nodes = set(b)

    # Find strongly connected components
    sccs = tarjan_scc(graph)

    # Count critical wires
    critical_wires = 0
    for scc in sccs:
        has_a = False
        has_b = False
        for node in scc:
            if node in a_nodes:
                has_a = True
            if node in b_nodes:
                has_b = True
        if has_a and has_b:
            continue
        num_cuts = 0
        for node in scc:
            if node not in a_nodes and node not in b_nodes:
                num_cuts += 1
        if num_cuts > 1:
            critical_wires += num_cuts

    return critical_wires

# Example usage
n = 5
m = 6
k = 2
l = 1
a = [1, 3]
b = [5]
edges = [(1, 2), (2, 3), (3, 1), (2, 4), (4, 5), (5, 2)]
print(count_critical_wires(n, m, k, l, a, b, edges))
