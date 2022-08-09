from contextlib import contextmanager

import rich.console
import varname

unique_memory = set()

console = rich.console.Console()


def dbg(*args, unique=None):
    if unique is None:
        pass
    elif unique in unique_memory:
        return
    else:
        unique_memory.add(unique)

    if len(args) == 1:
        console.print(varname.argname('args[0]', vars_only=False), "=", args[0])
    else:
        console.print(*args)


@contextmanager
def unique_debug(name):
    if name in unique_memory:
        with console.capture():
            yield
    else:
        console.print(f"Unique debug: {name}")
        yield
    unique_memory.add(name)
