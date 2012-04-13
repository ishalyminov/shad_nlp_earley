EXAMPLE_FILES = [
    'examples/task_1_positive',
    'examples/task_1_negative',
    'examples/task_2',
    'examples/task_4_positive',
    'examples/task_4_negative',
    'examples/task_5'
]

# returns a (header, examples) tuple
def load_examples(in_file):
    lines = [line.strip() for line in in_file]
    if len(lines) > 1:
        return (lines[0], lines[1:])
    return (None, None)