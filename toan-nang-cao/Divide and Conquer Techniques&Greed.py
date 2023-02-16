# binary search
def binary_search(arr, target):
	left = 0
	right = len(arr) - 1
	while left <= right:
		mid = (left + right) // 2
		if arr[mid] == target:
			return mid
		elif arr[mid] > target:
			right = mid - 1
		else:
			left = mid + 1
	return -1

# Quicksort
def quicksort(arr):
	if len(arr) < 2:
		return arr
	pivot = arr[0]
	less = [i for i in arr[1:] if i <= pivot]
	greater = [i for i in arr[1:] if i > pivot]
	return quicksort(less) + [pivot] + quicksort(greater)

# Kruskal
def kruskal(graph):
	# Sort edges by weight
	edges = sorted(graph.edges, key=lambda edge: edge.weight)
	# Create a set of vertices
	vertices = set()
	for edge in edges:
		vertices.add(edge.start)
		vertices.add(edge.end)
	# Create a list of disjoint sets
	disjoint_sets = []
	for vertex in vertices:
		disjoint_sets.append(DisjointSet(vertex))
	# Create a list of MST edges
	mst = []
	# For each edge in the graph
	for edge in edges:
		# Find the disjoint set of the start vertex
		start_set = None
		for disjoint_set in disjoint_sets:
			if disjoint_set.find(edge.start):
				start_set = disjoint_set
				break
		# Find the disjoint set of the end vertex
		end_set = None
		for disjoint_set in disjoint_sets:
			if disjoint_set.find(edge.end):
				end_set = disjoint_set
				break
		# If the start vertex and end vertex are not in the same set
		if start_set != end_set:
			# Add the edge to the MST
			mst.append(edge)
			# Union the two sets
			start_set.union(end_set)
	# Return the list of MST edges
	return mst

# Disjoint set
class DisjointSet:
	def __init__(self, data):
		self.data = data
		self.parent = self
		self.rank = 0

	def find(self, data):
		if self.data == data:
			return True
		if self.parent == self:
			return False
		return self.parent.find(data)

	def union(self, other):
		if self.rank > other.rank:
			other.parent = self
		else:
			self.parent = other
			if self.rank == other.rank:
				other.rank += 1

# Knapsack
def knapsack(items, capacity):
	# Create a table to store the maximum value at each capacity
	table = [0] * (capacity + 1)
	# For each item
	for item in items:
		# For each capacity in the table, in reverse order
		for c in range(capacity, 0, -1):
			# If the item's weight is less than or equal to the capacity
			if item.weight <= c:
				# Update the table value at that capacity
				table[c] = max(table[c], table[c - item.weight] + item.value)
	# Return the last value in the table
	return table[-1]

# main function
def main():
	# Define a graph
	class Graph:
		def __init__(self):
			self.edges = []
			self.vertices = set()
		def add_edge(self, start, end, weight):
			self.edges.append(Edge(start, end, weight))
			self.vertices.add(start)
			self.vertices.add(end)
	# Define an edge
	class Edge:
		def __init__(self, start, end, weight):
			self.start = start
			self.end = end
			self.weight = weight

	# Define an item
	class Item:
		def __init__(self, weight, value):
			self.weight = weight
			self.value = value

	graph = Graph()
	graph.edges = [Edge('A', 'B', 7), Edge('A', 'D', 5), Edge('B', 'C', 8), Edge('B', 'D', 9), Edge('B', 'E', 7), Edge('C', 'E', 5), Edge('D', 'E', 15), Edge('D', 'F', 6), Edge('E', 'F', 8)]

	# Test binary search
	arr = [1, 2, 3, 4, 5, 6]
	target = 4
	print(f"Binary Search : {binary_search(arr, target)}") 

	# Test quicksort
	arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
	print(f"Quicksort : {quicksort(arr)}")  

	# Test Kruskal
	mst = kruskal(graph)
	for edge in mst:
		print(f"{edge.start} - {edge.end} : {edge.weight}")

	# Test Knapsack
	items = [Item(3, 4), Item(4, 5), Item(2, 3), Item(1, 1), Item(5, 6)]
	capacity = 8
	print(f"Knapsack : {knapsack(items, capacity)}")  




if __name__ == '__main__':
	main()