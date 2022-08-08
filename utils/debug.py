import varname
from rich.console import Console

console = Console()


def dbg(*args):
    if len(args) == 1:
        console.print(varname.argname('args[0]', vars_only=False), "=", args[0])
    else:
        console.print(*args)
