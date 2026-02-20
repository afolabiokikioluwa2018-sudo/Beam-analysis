"""
Example structures and load cases for quick testing
"""

import pandas as pd

def load_example(example_name):
    """
    Load predefined example structures
    
    Returns:
        nodes_df, members_df, supports_df, loads_df
    """
    
    if example_name == "Simply Supported Beam":
        # Simple 6m beam with UDL
        nodes = pd.DataFrame({
            'Node': [1, 2],
            'X': [0.0, 6.0],
            'Y': [0.0, 0.0]
        })
        
        members = pd.DataFrame({
            'Member': [1],
            'Node_I': [1],
            'Node_J': [2],
            'E': [200e6],
            'I': [0.001],
            'A': [0.01]
        })
        
        supports = pd.DataFrame({
            'Node': [1, 2],
            'Type': ['Pinned', 'Roller'],
            'Dx': [0.0, 0.0],
            'Dy': [0.0, 0.0],
            'Rotation': [0.0, 0.0]
        })
        
        loads = pd.DataFrame({
            'Member': [1],
            'Type': ['UDL'],
            'W1': [10.0],
            'W2': [0.0],
            'P': [0.0],
            'M': [0.0],
            'a': [0.0],
            'b': [0.0]
        })
    
    elif example_name == "Continuous Beam - Multi Span":
        # Three-span continuous beam - demonstrating multi-span capability
        nodes = pd.DataFrame({
            'Node': [1, 2, 3, 4],
            'X': [0.0, 4.0, 8.0, 12.0],
            'Y': [0.0, 0.0, 0.0, 0.0]
        })
        
        members = pd.DataFrame({
            'Member': [1, 2, 3],
            'Node_I': [1, 2, 3],
            'Node_J': [2, 3, 4],
            'E': [200e6, 200e6, 200e6],
            'I': [0.001, 0.001, 0.001],
            'A': [0.01, 0.01, 0.01]
        })
        
        supports = pd.DataFrame({
            'Node': [1, 2, 3, 4],
            'Type': ['Pinned', 'Roller', 'Roller', 'Roller'],
            'Dx': [0.0, 0.0, 0.0, 0.0],
            'Dy': [0.0, 0.0, 0.0, 0.0],
            'Rotation': [0.0, 0.0, 0.0, 0.0]
        })
        
        loads = pd.DataFrame({
            'Member': [1, 2, 3],
            'Type': ['UDL', 'Point Load', 'UDL'],
            'W1': [15.0, 0.0, 12.0],
            'W2': [0.0, 0.0, 0.0],
            'P': [0.0, 50.0, 0.0],
            'M': [0.0, 0.0, 0.0],
            'a': [0.0, 2.0, 0.0],
            'b': [0.0, 0.0, 0.0]
        })
    
    elif example_name == "Portal Frame":
        # Simple portal frame - single storey
        nodes = pd.DataFrame({
            'Node': [1, 2, 3, 4],
            'X': [0.0, 0.0, 6.0, 6.0],
            'Y': [0.0, 4.0, 4.0, 0.0]
        })
        
        members = pd.DataFrame({
            'Member': [1, 2, 3],
            'Node_I': [1, 2, 3],
            'Node_J': [2, 3, 4],
            'E': [200e6, 200e6, 200e6],
            'I': [0.002, 0.0015, 0.002],
            'A': [0.015, 0.012, 0.015]
        })
        
        supports = pd.DataFrame({
            'Node': [1, 4],
            'Type': ['Fixed', 'Fixed'],
            'Dx': [0.0, 0.0],
            'Dy': [0.0, 0.0],
            'Rotation': [0.0, 0.0]
        })
        
        loads = pd.DataFrame({
            'Member': [2, 2],
            'Type': ['UDL', 'Point Load'],
            'W1': [8.0, 0.0],
            'W2': [0.0, 0.0],
            'P': [0.0, 20.0],
            'M': [0.0, 0.0],
            'a': [0.0, 0.0],
            'b': [0.0, 0.0]
        })
    
    elif example_name == "Multi-Storey Frame":
        # Two-storey frame demonstrating vertical and horizontal members
        nodes = pd.DataFrame({
            'Node': [1, 2, 3, 4, 5, 6],
            'X': [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
            'Y': [0.0, 3.0, 6.0, 6.0, 3.0, 0.0]
        })
        
        members = pd.DataFrame({
            'Member': [1, 2, 3, 4, 5, 6],
            'Node_I': [1, 2, 2, 3, 5, 6],
            'Node_J': [2, 3, 5, 4, 6, 1],  # Note: Member 6 creates diagonal
            'E': [200e6, 200e6, 200e6, 200e6, 200e6, 200e6],
            'I': [0.002, 0.002, 0.0015, 0.0015, 0.002, 0.002],
            'A': [0.015, 0.015, 0.012, 0.012, 0.015, 0.015]
        })
        
        supports = pd.DataFrame({
            'Node': [1, 6],
            'Type': ['Fixed', 'Fixed'],
            'Dx': [0.0, 0.0],
            'Dy': [0.0, 0.0],
            'Rotation': [0.0, 0.0]
        })
        
        loads = pd.DataFrame({
            'Member': [3, 4, 2],
            'Type': ['UDL', 'UDL', 'Point Load'],
            'W1': [10.0, 10.0, 0.0],
            'W2': [0.0, 0.0, 0.0],
            'P': [0.0, 0.0, 30.0],
            'M': [0.0, 0.0, 0.0],
            'a': [0.0, 0.0, 1.5],
            'b': [0.0, 0.0, 0.0]
        })
    
    elif example_name == "Frame with Settlement":
        # Two-span beam with support settlement
        nodes = pd.DataFrame({
            'Node': [1, 2, 3],
            'X': [0.0, 5.0, 10.0],
            'Y': [0.0, 0.0, 0.0]
        })
        
        members = pd.DataFrame({
            'Member': [1, 2],
            'Node_I': [1, 2],
            'Node_J': [2, 3],
            'E': [200e6, 200e6],
            'I': [0.001, 0.001],
            'A': [0.01, 0.01]
        })
        
        supports = pd.DataFrame({
            'Node': [1, 2, 3],
            'Type': ['Fixed', 'Roller', 'Roller'],
            'Dx': [0.0, 0.0, 0.0],
            'Dy': [0.0, -0.02, 0.0],  # 20mm settlement at middle support
            'Rotation': [0.0, 0.0, 0.0]
        })
        
        loads = pd.DataFrame({
            'Member': [1, 2],
            'Type': ['UDL', 'UDL'],
            'W1': [20.0, 20.0],
            'W2': [0.0, 0.0],
            'P': [0.0, 0.0],
            'M': [0.0, 0.0],
            'a': [0.0, 0.0],
            'b': [0.0, 0.0]
        })
    
    else:
        # Return empty dataframes
        nodes = pd.DataFrame(columns=['Node', 'X', 'Y'])
        members = pd.DataFrame(columns=['Member', 'Node_I', 'Node_J', 'E', 'I', 'A'])
        supports = pd.DataFrame(columns=['Node', 'Type', 'Dx', 'Dy', 'Rotation'])
        loads = pd.DataFrame(columns=['Member', 'Type', 'W1', 'W2', 'P', 'M', 'a', 'b'])
    
    return nodes, members, supports, loads
