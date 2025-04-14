import yaml
import sys
from pathlib import Path


def set_nested_value(d, path, value):
    """Set a value in a nested dictionary using a dot-separated path.
    
    Args:
        d: The dictionary to modify
        path: Dot-separated path to the value (e.g., 'trainer.learning_rate')
        value: The value to set
        
    Returns:
        The modified dictionary
    """
    keys = path.split('.')
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    
    # Convert value to appropriate type
    if value.lower() == 'true':
        value = True
    elif value.lower() == 'false':
        value = False
    elif value.lower() == 'null':
        value = None
    else:
        # Try to convert to number if possible
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
    
    d[keys[-1]] = value
    return d


def adjust_config(config_path, modifications):
    """Modify a YAML configuration file with the specified modifications.
    
    Args:
        config_path: Path to the original configuration file
        modifications: Dictionary of parameter paths and new values
        output_path: Path to save the modified configuration (if None, overwrites original)
        
    Returns:
        True if successful, False otherwise
    """
    # Load the configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return [set_nested_value(config, param_path, new_value) for param_path, new_value in modifications.items()]
    except Exception as e:
        print(f"Error adjusting configuration: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage from command line
    if len(sys.argv) < 3:
        print("Usage: python adjust_config.py <config_path> <output_path> <param1=value1> [param2=value2 ...]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Parse modifications from command line arguments
    modifications = {}
    for arg in sys.argv[3:]:
        if '=' in arg:
            param, value = arg.split('=', 1)
            modifications[param] = value
    
    # Adjust the configuration
    success = adjust_config(config_path, modifications, output_path)
    sys.exit(0 if success else 1)
