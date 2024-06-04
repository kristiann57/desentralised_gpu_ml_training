package registry

import (
	"sync"
	"time"
)

// Registry is a struct that holds a map of all the available GPUs.

type NodeRegistry struct {
	mu    sync.Mutex
	nodes map[string]*NodeDetails
}

// Updated NodeDetails to include lastCheckIn and load
type NodeDetails struct {
	id          string
	isActive    bool
	tasks       int
	lastCheckIn time.Time
	load        int // An example load metric
}

// --------------------- NODE STATUS --------------------- ///

// UpdateNodeLoad updates the load and check-in time for a registered node.
// This is a new function to handle dynamic load balancing.
func (r *NodeRegistry) UpdateNodeLoad(id string, load int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if node, exists := r.nodes[id]; exists {
		node.load = load
		node.lastCheckIn = time.Now() // Update the check-in time to the current time
	}
}

// --------------------- NODE REGISTRATION --------------------- ///

func (r *NodeRegistry) RegisterNode(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nodes[id] = &NodeDetails{id: id, isActive: true, tasks: 0}
}

func (r *NodeRegistry) UnregisterNode(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.nodes, id)
}

// --------------------- NODE ALLOCATION --------------------- ///

// AllocateNode selects the least busy active node for task allocation.
// This function has been updated to utilize the enhanced node status information.
func (r *NodeRegistry) AllocateNode() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var selectedNodeID string
	minLoad := int(^uint(0) >> 1) // Initialize to maximum int value

	for id, node := range r.nodes {
		if node.isActive && node.lastCheckIn.Add(5*time.Minute).After(time.Now()) { // Check node is active and checked in within last 5 minutes
			if node.load < minLoad {
				selectedNodeID = id
				minLoad = node.load
			}
		}
	}
	if selectedNodeID != "" {
		r.nodes[selectedNodeID].tasks++ // Increment task count for the selected node
	}
	return selectedNodeID
}

// NewRegistry creates a new instance of a registry.
func NewRegistry() *NodeRegistry {
	return &NodeRegistry{
		nodes: make(map[string]*NodeDetails), // make a map of clients.
	}
}
