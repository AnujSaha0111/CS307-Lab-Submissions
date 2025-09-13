def get_literal_value(lit, assignment):
    var = abs(lit)
    val = assignment[var]
    if lit < 0:
        val = not val
    return val