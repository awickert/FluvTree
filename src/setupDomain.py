import pandas as pd
import networkx as nx
import numpy as np
import sys
from matplotlib import pyplot as plt

# Convert each of the sheets of the spreadsheet to a CSV for import
# (direct spreadsheet import is not possible with lists in cells)

# Convert spreadsheet to CSVs
# Hard-coded for this test
xlsx = pd.ExcelFile('../example-data/RiverDiGraphTest.xlsx')
df_edges = pd.read_excel(xlsx, 'streams')
df_nodes = pd.read_excel(xlsx, 'nodes')

# I suppsoe that I could convert the format of those columns, but I will not.
# At least for now
df_edges.to_csv('RiverDiGraphTest_streams.csv', index=False)
df_nodes.to_csv('RiverDiGraphTest_nodes.csv', index=False)

def strlist_to_floatarray(a):
    _a = a.strip("[]")
    if len(_a) == 0:
        return []
    else:
        return np.array(list(map(float, _a.split(','))))

def strlist_to_intlist(a):
    _a = a.strip("[]")
    if len(_a) == 0:
        return []
    else:
        return list(map(int, _a.split(',')))

# Impport DataFrames
df_edges = pd.read_csv('RiverDiGraphTest_streams.csv',
                       converters={
                           'x': lambda a: strlist_to_floatarray(a),
                           'y': lambda a: strlist_to_floatarray(a),
                           'z': lambda a: strlist_to_floatarray(a),
                           'Q': lambda a: strlist_to_floatarray(a)
                           },
                        index_col='edge_id'
                        )
df_nodes = pd.read_csv('RiverDiGraphTest_nodes.csv',
                       converters={
                           'from_edges': lambda a: strlist_to_intlist(a),
                           'to_edges': lambda a: strlist_to_intlist(a),
                           },
                        index_col='node_id'
                        )

# Build the extra information needed to run the calculations
_ds_upstream_list = [] # using "s" for "along horizontal curve"
_dz_upstream_list = []
_ds_downstream_list = []
_dz_downstream_list = []
_dQ_upstream_list = []
_dQ_downstream_list = []
for i in range(len(df_edges)):
    segment = df_edges.iloc[i]
    # Create extended lists by combining information at nodes with the
    # arrays from the river segments
    _x = np.hstack((    [ df_nodes.iloc[segment['upstream_node']]['x'] ],
                        segment['x'],
                        [ df_nodes.iloc[segment['downstream_node']]['x'] ]
                    ))
    _y = np.hstack((    [ df_nodes.iloc[segment['upstream_node']]['y'] ],
                        segment['y'],
                        [ df_nodes.iloc[segment['downstream_node']]['y'] ]
                    ))
    _z = np.hstack((    [ df_nodes.iloc[segment['upstream_node']]['z'] ],
                        segment['z'],
                        [ df_nodes.iloc[segment['downstream_node']]['z'] ]
                    ))
    _Q = np.hstack((    [ df_nodes.iloc[segment['upstream_node']]['Q'] ],
                        segment['Q'],
                        [ df_nodes.iloc[segment['downstream_node']]['Q'] ]
                    ))
    # Then for horizontal change, use the distance formula
    ds = ( np.diff(_x)**2 + np.diff(_y)**2 ) **.5
    # But for other variables, a simple difference will do
    dz = np.diff( _z )
    dQ = np.diff( _Q )
    # And append these to the lists
    _ds_upstream_list.append( ds[:-1] )
    _ds_downstream_list.append( ds[1:] )
    _dz_upstream_list.append( dz[:-1] )
    _dz_downstream_list.append( dz[1:] )
    _dQ_upstream_list.append( dQ[1:] )
    _dQ_downstream_list.append( dQ[1:] )

# Once this is done, we can create and populate new columns
df_edges['ds_upstream'] = _ds_upstream_list
df_edges['ds_downstream'] = _ds_downstream_list
df_edges['dz_upstream'] = _dz_upstream_list
df_edges['dz_downstream'] = _dz_downstream_list

# Generate network structure with data on edges
G = nx.from_pandas_edgelist(df_edges, source='upstream_node', target='downstream_node', edge_key='edge_id', edge_attr=True, create_using=nx.DiGraph)

# Add data on nodes
G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())



# Now, to see if it worked: try to plot!

# To do this, assume we will have an acyclic binary tree and walk up it
# We start by defining the number of the mouth node; this could later
# set externally
mouth_node = 5
s = nx.bfs_tree(G, mouth_node, reverse=True)
# We start by placing distance = 0 at the mouth node
G.nodes[mouth_node]['s'] = 0

# Loop to get all upstream distances
for node in s.nodes:
    # First, update the distance upstream from the node
    if node == mouth_node:
        # I never like having a single-case "if". seems like a big waste.
        # Ways around this?
        G.nodes[node]['s'] = 0.
        # Could also check if out_edges is nonexistent...
    else:
        # Otherwise, look at the next link downstream
        edges = G.out_edges(node)
        if len(edges) != 1:
            # Declare an error if we don't have exactly 1 downstream edge
            # This is built for only directed, convergent, acyclic graphs
            sys.exit() # error
        else:
            edge = next(iter(edges))
        # First item = upstream
        _dx = G.nodes[node]['x'] - G.edges[edge]['x'][0]
        _dy = G.nodes[node]['y'] - G.edges[edge]['y'][0]
        ds = ( _dx**2 + _dy**2 )**.5
        # Update the node
        G.nodes[node]['s'] = ds + G.edges[edge]['s'][0]
    # Then walk upstream to the edges touching this node and iterate
    # over them.
    # For a tributary network, this should be 1 for the mouth node, 2
    # for all middle nodes, and 0 for the upstream-most-end nodes
    edges_to_node = G.in_edges(node)
    for edge in edges_to_node:
        # Find distances upstream from next downstream node
        # I have set the convention that x,y,z are arranged
        # upstream to downstream
        _x = np.hstack(( G.edges[edge]['x'], [G.nodes[node]['x']] )) 
        _y = np.hstack(( G.edges[edge]['y'], [G.nodes[node]['y']] )) 
        # Continue without assuming that we've already calculated dx, dy, ds
        ds = ( np.diff(_x)**2 + np.diff(_y)**2 )**.5
        # Increasing (positive) distance with distance from the river mouth 
        G.edges[edge]['s']= np.cumsum(ds[::-1])[::-1] + G.nodes[node]['s']
    

# Try plotting
# Color each of the segments, up to and including the nodes above them.
# Use gray for the downstream connectors

# Because of this plan, we can start by simply looping over nodes and seeing
# what is downstream of each of them
s_on_segment_list = []
var_on_segment_list = []
varname = 'z' # <-- should be able to alter how we obtain this
for node in G.nodes:
    # By this point, just assume binary tree
    edges = G.out_edges(node)
    if len(edges) == 0:
        break # we are at the mouth -- this should happen only once.
    elif len(edges) > 1:
        sys.exit() # something bad has happened
    else:
        edge = next(iter( edges ))
    _s = np.hstack(( [G.nodes[node]['s']], G.edges[edge]['s'] ))
    _var = np.hstack(( [G.nodes[node][varname]], G.edges[edge][varname] ))
    s_on_segment_list.append(_s)
    var_on_segment_list.append(_var)

# Then we can go through nodes and find what is upstream of them
s_on_node_list = []
var_on_node_list = []
varname = 'z' # <-- should be able to alter how we obtain this
for node in G.nodes:
    # Let's see what's coming in
    edges = G.in_edges(node)
    if len(edges) == 0:
        continue # upstream-most nodes; these are already included in segments
    else:
        # Here, should have multiple connections
        for edge in edges:
            _s = np.hstack(( [G.edges[edge]['s'][-1]], [G.nodes[node]['s']] ))
            _var = np.hstack(( [G.edges[edge][varname][-1]],
                                [G.nodes[node][varname]] ))
            s_on_node_list.append(_s)
            var_on_node_list.append(_var)

#plt.ion()
plt.figure()
for i in range(len(s_on_node_list)):
    plt.plot( s_on_node_list[i], var_on_node_list[i], 'k-', linewidth=3,
                alpha=0.3 )
for i in range(len(s_on_segment_list)):
    plt.plot( s_on_segment_list[i], var_on_segment_list[i], 'k-',
                linewidth=3, alpha=1 )

plt.xlabel('Upstream distance')
plt.ylabel('Elevation') # hard-coded for now

plt.show()

#    print( node )
#    _dx = G.nodes[node]['x'] - G.nodes[previous_point]['x']
#    _
    
