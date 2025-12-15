def split_string_to_prefix_list(state_str, max_prefixes=10):
    """
    Split a string into a list of progressive prefixes based on:
    1. Function endings (dedented lines after function body)
    2. Docstring/comment endings (''' or \"\"\")

    Args:
        state_str: The code string to split
        max_prefixes: Maximum number of prefixes to return (default: 10)

    Returns:
        List of prefixes, with the last one always being the complete string
    """
    lines = state_str.split('\n')
    prefixes = []
    current_prefix = []

    in_triple_single = False  # for ''' '''
    in_triple_double = False  # for """ """
    prev_indent = 0
    in_function = False

    for i, line in enumerate(lines):
        current_prefix.append(line)
        stripped = line.strip()

        # Track indentation
        if stripped:
            current_indent = len(line) - len(line.lstrip())
        else:
            current_indent = prev_indent

        # Check for function start
        if stripped.startswith('def ') or stripped.startswith('async def '):
            in_function = True
            prev_indent = current_indent
            continue

        # Check for triple quote toggles
        # Count occurrences in the line
        triple_single_count = line.count("'''")
        triple_double_count = line.count('"""')

        # Toggle triple single quote state
        if triple_single_count % 2 == 1:
            in_triple_single = not in_triple_single
            # If we just closed a triple quote, create a prefix
            if not in_triple_single and len(prefixes) < max_prefixes - 1:
                prefixes.append('\n'.join(current_prefix))

        # Toggle triple double quote state
        if triple_double_count % 2 == 1:
            in_triple_double = not in_triple_double
            # If we just closed a triple quote, create a prefix
            if not in_triple_double and len(prefixes) < max_prefixes - 1:
                prefixes.append('\n'.join(current_prefix))

        # Check for function end (dedent after function body)
        if in_function and stripped and not in_triple_single and not in_triple_double:
            if current_indent <= prev_indent and i > 0:
                # We've dedented, meaning the function likely ended
                # Look back to see if previous line was part of function
                if lines[i-1].strip():  # Previous line wasn't empty
                    in_function = False
                    if len(prefixes) < max_prefixes - 1:
                        # Add prefix up to the previous line (end of function)
                        prefixes.append('\n'.join(current_prefix[:-1]))

        prev_indent = current_indent

    # Always add the complete string as the last prefix
    complete_str = '\n'.join(current_prefix)
    if not prefixes or prefixes[-1] != complete_str:
        prefixes.append(complete_str)

    # Ensure we don't exceed max_prefixes
    if len(prefixes) > max_prefixes:
        # Keep first (max_prefixes - 1) and the complete string
        prefixes = prefixes[:max_prefixes - 1] + [complete_str]

    return prefixes
