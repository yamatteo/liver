def range_s(total: int, step: int, min: int = None):
    if min and total % step < min:
        raise ValueError(f"Step {step} and total {total} give last piece of length {total % step} < min = {min}.")
    yield from range(0, total, step)

def range_q(total: int, quantity: int):
    step = (total + quantity - 1) // quantity
    yield from range(0, total, step)

def range_o(total:int, step:int):
    quantity = (total + step - 1) // step
    if quantity < 2:
        yield 0
        return
    for i in range(quantity):
        yield i*(total-step)//(quantity-1)
