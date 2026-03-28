# utils.py
# Utility functions for debugging
DEBUG_MODE = True

# ANSI Escape Codes for Colors
CLR_DEBUG = '\x1b[36m'    # Blue
CLR_RESET = "\033[0m"     # Reset to default

def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        # We join all arguments with a space, wrap in blue, and reset at the end
        # Using 'sep' from kwargs if provided, otherwise default to space
        sep = kwargs.get('sep', ' ')
        message = sep.join(map(str, args))
        print(f"{CLR_DEBUG}{message}{CLR_RESET}", **{k: v for k, v in kwargs.items() if k != 'sep'})
        
def set_debug(state: bool):
    """Function to toggle debug mode from other files"""
    global DEBUG_MODE
    DEBUG_MODE = state